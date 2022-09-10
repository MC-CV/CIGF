import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from DataHandler import negSamp, transpose, DataHandler, transToLsts
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
import scipy.sparse as sp
from print_hook import PrintHook
import datetime
from time import time

class Recommender:
    def __init__(self, sess, handler):
        self.sess = sess
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        self.metrics = dict()
        self.weights = self._init_weights()
        self.behEmbeds = NNs.defineParam('behEmbeds', [args.behNum, args.latdim // 2])
        if args.data == 'beibei':
            mets = ['Loss', 'preLoss', 'HR', 'NDCG', 'HR45', 'NDCG45', 'HR50', 'NDCG50', 'HR55', 'NDCG55', 'HR60', 'NDCG60', 'HR65', 'NDCG65', 'HR100', 'NDCG100']
        else:
            mets = ['Loss', 'preLoss', 'HR', 'NDCG', 'HR20', 'NDCG20', 'HR25', 'NDCG25', 'HR30', 'NDCG30', 'HR35', 'NDCG35', 'HR100', 'NDCG100']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            init = tf.global_variables_initializer()
            self.sess.run(init)
            log('Variables Inited')
        train_time = 0
        test_time = 0
        for ep in range(stloc, args.epoch):
            test = (ep % args.tstEpoch == 0)
            t0 = time()
            reses = self.trainEpoch()
            t1 = time()
            train_time += t1-t0
            print('Train_time',t1-t0,'Total_time',train_time)
            log(self.makePrint('Train', ep, reses, test))
            if test:                  
                t2 = time()
                reses = self.testEpoch()
                t3 = time()
                test_time += t3-t2
                print('Test_time',t3-t2,'Total_time',test_time)
                log(self.makePrint('Test', ep, reses, test))                                                                    
            if ep % args.tstEpoch == 0:
                self.saveHistory()
            print()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        self.saveHistory()


    def messagePropagate(self, lats, adj):
        return Activate(tf.sparse_tensor_dense_matmul(adj, lats), self.actFunc)


    def defineModel(self):
        uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim // 2], reg=True)
        iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim // 2], reg=True)
        allEmbed = tf.concat([uEmbed0, iEmbed0], axis = 0)

        self.ulat = [0] * (args.behNum)
        self.ilat = [0] * (args.behNum)
        for beh in range(args.behNum):
            ego_embeddings = allEmbed
            all_embeddings = [ego_embeddings]
            if args.multi_graph == False:
                for index in range(args.gnn_layer):
                    symm_embeddings = tf.sparse_tensor_dense_matmul(self.adjs[beh], all_embeddings[-1])
                    if args.encoder == 'lightgcn':
                        lightgcn_embeddings = symm_embeddings
                        all_embeddings.append(lightgcn_embeddings)
                    elif args.encoder == 'gccf':
                        gccf_embeddings = Activate(symm_embeddings, self.actFunc)
                        all_embeddings.append(gccf_embeddings)
                    elif args.encoder == 'gcn':
                        gcn_embeddings = Activate(
                            tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights[
                                'b_gc_%d' % index], self.actFunc)
                        all_embeddings.append(gcn_embeddings)
                    elif args.encoder == 'ngcf':
                        gcn_embeddings = Activate(
                            tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights[
                                'b_gc_%d' % index], self.actFunc)
                        bi_embeddings = tf.multiply(ego_embeddings, gcn_embeddings)
                        bi_embeddings = Activate(
                            tf.matmul(bi_embeddings, self.weights['W_bi_%d' % index]) + self.weights['b_bi_%d' % index],
                            self.actFunc)
                        all_embeddings.append(gcn_embeddings + bi_embeddings)

            elif args.multi_graph == True:
                for index in range(args.gnn_layer):
                    if index == 0:
                        symm_embeddings = tf.sparse_tensor_dense_matmul(self.adjs[beh], all_embeddings[-1])
                        if args.encoder == 'lightgcn':
                            lightgcn_embeddings = symm_embeddings
                            all_embeddings.append(lightgcn_embeddings + all_embeddings[-1])
                        elif args.encoder == 'gccf':
                            gccf_embeddings = Activate(symm_embeddings, self.actFunc)
                            all_embeddings.append(gccf_embeddings + all_embeddings[-1])
                        elif args.encoder == 'gcn':
                            gcn_embeddings = Activate(tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights['b_gc_%d' % index], self.actFunc)
                            all_embeddings.append(gcn_embeddings + all_embeddings[-1])
                        elif args.encoder == 'ngcf':
                            gcn_embeddings = Activate(tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights['b_gc_%d' % index], self.actFunc)
                            bi_embeddings = tf.multiply(ego_embeddings, gcn_embeddings)
                            bi_embeddings = Activate(tf.matmul(bi_embeddings, self.weights['W_bi_%d' % index]) + self.weights['b_bi_%d' % index], self.actFunc)
                            all_embeddings.append(gcn_embeddings + bi_embeddings + all_embeddings[-1])
                    else:
                        atten = FC(ego_embeddings, args.behNum, reg=True, useBias=True,
                                   activation=self.actFunc, name='attention_%d_%d'%(beh,index), reuse=True)
                        temp_embeddings = []
                        for inner_beh in range(args.behNum):
                            neighbor_embeddings = tf.sparse_tensor_dense_matmul(self.adjs[inner_beh], symm_embeddings)
                            temp_embeddings.append(neighbor_embeddings)
                        all_temp_embeddings = tf.stack(temp_embeddings, 1)
                        symm_embeddings = tf.reduce_sum(tf.einsum('abc,ab->abc', all_temp_embeddings, atten), axis=1, keepdims=False)
                        if args.encoder == 'lightgcn':
                            lightgcn_embeddings = symm_embeddings
                            all_embeddings.append(lightgcn_embeddings + all_embeddings[-1])
                        elif args.encoder == 'gccf':
                            gccf_embeddings = Activate(symm_embeddings, self.actFunc)
                            all_embeddings.append(gccf_embeddings + all_embeddings[-1])
                        elif args.encoder == 'gcn':
                            gcn_embeddings = Activate(tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights['b_gc_%d' % index], self.actFunc)
                            all_embeddings.append(gcn_embeddings + all_embeddings[-1])
                        elif args.encoder == 'ngcf':
                            gcn_embeddings = Activate(tf.matmul(symm_embeddings, self.weights['W_gc_%d' % index]) + self.weights['b_gc_%d' % index], self.actFunc)
                            bi_embeddings = tf.multiply(ego_embeddings, gcn_embeddings)
                            bi_embeddings = Activate(tf.matmul(bi_embeddings, self.weights['W_bi_%d' % index]) + self.weights['b_bi_%d' % index], self.actFunc)
                            all_embeddings.append(gcn_embeddings + bi_embeddings + all_embeddings[-1])

            all_embeddings = tf.add_n(all_embeddings)
            self.ulat[beh], self.ilat[beh] = tf.split(all_embeddings, [args.user, args.item], 0)
        self.ulat_merge, self.ilat_merge = tf.add_n(self.ulat), tf.add_n(self.ilat)


    def _init_weights(self):
        all_weights = dict()
        initializer = tf.random_normal_initializer(stddev=0.01)  

        self.weight_size_list = [args.latdim // 2] + [args.latdim // 2] * args.gnn_layer

        for k in range(args.gnn_layer):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_mlp_%d' % k)

        return all_weights

    def bilinear_predict(self, src):
        uids = self.uids[src]
        iids = self.iids[src]

        src_ulat = tf.nn.embedding_lookup(self.ulat[src], uids)
        src_ilat = tf.nn.embedding_lookup(self.ilat[src], iids)

        behEmbed = self.behEmbeds[src]
        predEmbed = tf.reduce_sum(src_ulat * src_ilat * tf.expand_dims(behEmbed, axis=0), axis=-1, keep_dims=False)
        preds = predEmbed

        return preds * args.mult

    def shared_bottom_predict(self, src):
        uids = self.uids[src]
        iids = self.iids[src]

        src_ulat = tf.nn.embedding_lookup(self.ulat_merge, uids)
        src_ilat = tf.nn.embedding_lookup(self.ilat_merge, iids)

        preds = tf.squeeze(FC(tf.concat([src_ulat,src_ilat], axis=-1), 1, reg=True, useBias=True,
                     name='tower_' + str(src), reuse=True))

        return preds * args.mult

    def mmoe_predict(self, src):
        uids = self.uids[src]
        iids = self.iids[src]

        src_ulat = tf.nn.embedding_lookup(self.ulat_merge, uids)
        src_ilat = tf.nn.embedding_lookup(self.ilat_merge, iids)

        exper_info = []
        for i in range(args.num_exps):
            exper_net = FC(tf.concat([src_ulat,src_ilat], axis=-1), args.latdim, reg=True, useBias=True,
                     activation=self.actFunc, name='expert_' + str(i), reuse=True)
            exper_info.append(exper_net)
        expert_concat = tf.stack(exper_info, axis = 1)

        gate_out = FC(tf.concat([src_ulat,src_ilat], axis=-1), args.num_exps, reg=True, useBias=True,
                    activation='softmax', name='gate_softmax_' + str(src), reuse=True)
        
        mmoe_out = tf.reduce_sum(tf.expand_dims(gate_out, axis = -1) * expert_concat, axis=1, keep_dims=False)

        preds = tf.squeeze(FC(mmoe_out, 1, reg=True, useBias=True,
                    name='tower_' + str(src), reuse=True))

        return preds * args.mult

    def ple_predict(self, src):
        uids = self.uids[src]
        iids = self.iids[src]

        src_ulat = tf.nn.embedding_lookup(self.ulat_merge, uids)
        src_ilat = tf.nn.embedding_lookup(self.ilat_merge, iids)

        def cgc_net(level_name):
            specific_expert_outputs = []
            if args.num_exps == 3:
                specific_expert_num = 1
                shared_expert_num = 1
            else:
                specific_expert_num = 2
                shared_expert_num = 1
            for i in range(specific_expert_num):
                expert_network = FC(tf.concat([src_ulat, src_ilat], axis=-1), args.latdim, reg=True, useBias=True,
                                    activation=self.actFunc, name=level_name + '_expert_specific_' + str(i) + str(src),
                                    reuse=True)
                specific_expert_outputs.append(expert_network)
            shared_expert_outputs = []
            for k in range(shared_expert_num):
                expert_network = FC(tf.concat([src_ulat, src_ilat], axis=-1), args.latdim, reg=True, useBias=True,
                                    activation=self.actFunc, name=level_name + 'expert_shared_' + str(k), reuse=True)
                shared_expert_outputs.append(expert_network)

            cur_expert_num = specific_expert_num + shared_expert_num
            cur_experts = specific_expert_outputs + shared_expert_outputs

            expert_concat = tf.stack(cur_experts, axis=1)

            gate_out = FC(tf.concat([src_ulat, src_ilat], axis=-1), cur_expert_num, reg=True, useBias=True,
                          activation='softmax', name='gate_softmax_' + str(src), reuse=True)
            gate_out = tf.expand_dims(gate_out, axis=-1)

            gate_mul_expert = tf.reduce_sum(expert_concat * gate_out, axis=1, keep_dims=False)
            return gate_mul_expert

        ple_outputs = cgc_net(level_name='level_')
        preds = tf.squeeze(FC(ple_outputs, 1, reg=True, useBias=True,
                    name='tower_' + str(src), reuse=True))
        return preds * args.mult

    def sesg_predict(self, src):
        uids = self.uids[src]
        iids = self.iids[src]

        src_ulat = tf.nn.embedding_lookup(self.ulat[src], uids)
        src_ilat = tf.nn.embedding_lookup(self.ilat[src], iids)

        metalat111 = FC(tf.concat([src_ulat, src_ilat], axis=-1), args.behNum, reg=True, useBias=True,
                        activation='softmax', name='gate111', reuse=True)
        w1 = tf.reshape(metalat111, [-1, args.behNum, 1])
        
        exper_info = []
        for index in range(args.behNum):
            exper_info.append(
                tf.nn.embedding_lookup(self.ulat[index], uids) * tf.nn.embedding_lookup(self.ilat[index], iids))
        predEmbed = tf.stack(exper_info, axis=2)
        sesg_out = tf.reshape(predEmbed @ w1, [-1, args.latdim // 2])

        preds = tf.squeeze(tf.reduce_sum(sesg_out, axis=-1))

        return preds * args.mult

    def create_multiple_adj_mat(self, adj_mat):
        def left_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate left_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        def right_adj_single(adj):
            rowsum = np.array(adj.sum(0))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = adj.dot(d_mat_inv)
            print('generate right_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        def symm_adj_single(adj_mat):
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            rowsum = np.array(adj_mat.sum(0))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv_trans = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv_trans)
            print('generate symm_adj_single adjacency matrix.')
            return norm_adj.tocoo()

        left_adj_mat = left_adj_single(adj_mat)
        right_adj_mat = right_adj_single(adj_mat)
        symm_adj_mat = symm_adj_single(adj_mat)

        return left_adj_mat.tocsr(), right_adj_mat.tocsr(), symm_adj_mat.tocsr()

    def mult(self,a,b):
        return (a*b).sum(1)
    
    def coef(self,buy_mat,view_mat):
        buy_dense = np.array(buy_mat.todense())
        view_dense = np.array(view_mat.todense())
        buy = buy_dense-buy_dense.sum(1).reshape(-1,1)/buy_dense.shape[1]
        view = view_dense-view_dense.sum(1).reshape(-1,1)/view_dense.shape[1]
        return self.mult(buy,view)/np.sqrt((self.mult(buy,buy))*self.mult(view,view))
      
    def mult_tmall(self,a,b):
        return a.multiply(b).sum(1)
    
    def coef_tmall(self,buy_mat,view_mat):
        buy = buy_mat
        view = view_mat
        return np.array(self.mult_tmall(buy,view))/np.sqrt(np.array(self.mult_tmall(buy,buy))*np.array(self.mult_tmall(view,view)))
    
    def prepareModel(self):
        self.actFunc = 'leakyRelu'
        self.adjs = []
        self.uids, self.iids = [], []
        self.left_trnMats, self.right_trnMats, self.symm_trnMats, self.none_trnMats = [], [], [], []

        for i in range(args.behNum):
            R = self.handler.trnMats[i].tolil()
            
            coomat = sp.coo_matrix(R)
            coomat_t = sp.coo_matrix(R.T)
            row = np.concatenate([coomat.row, coomat_t.row + R.shape[0]])
            col = np.concatenate([R.shape[0] + coomat.col, coomat_t.col])
            data = np.concatenate([coomat.data.astype(np.float32), coomat_t.data.astype(np.float32)])
            adj_mat = sp.coo_matrix((data, (row, col)), shape=(args.user + args.item, args.user + args.item))

            
            left_trn, right_trn, symm_trn = self.create_multiple_adj_mat(adj_mat)
            self.left_trnMats.append(left_trn)
            self.right_trnMats.append(right_trn)
            self.symm_trnMats.append(symm_trn)
            self.none_trnMats.append(adj_mat.tocsr())
        if args.normalization == "left":
            self.final_trnMats = self.left_trnMats
        elif args.normalization == "right":
            self.final_trnMats = self.right_trnMats
        elif args.normalization == "symm":
            self.final_trnMats = self.symm_trnMats
        elif args.normalization == 'none':
            self.final_trnMats = self.none_trnMats

        for i in range(args.behNum):
            adj = self.final_trnMats[i]
            idx, data, shape = transToLsts(adj, norm=False)
            self.adjs.append(tf.sparse.SparseTensor(idx, data, shape))
            self.uids.append(tf.placeholder(name='uids' + str(i), dtype=tf.int32, shape=[None]))
            self.iids.append(tf.placeholder(name='iids' + str(i), dtype=tf.int32, shape=[None]))


        self.defineModel()

        self.preLoss = 0
        for src in range(args.behNum):
            if args.decoder == 'single':
                if src != args.behNum-1:
                    continue
                preds = self.shared_bottom_predict(src)
            elif args.decoder == 'bilinear':
                preds = self.bilinear_predict(src)
            elif args.decoder == 'shared_bottom':
                preds = self.shared_bottom_predict(src)
            elif args.decoder == 'mmoe':
                preds = self.mmoe_predict(src)
            elif args.decoder == 'ple':
                preds = self.ple_predict(src)
            elif args.decoder == 'sesg':
                preds = self.sesg_predict(src)

            sampNum = tf.shape(self.uids[src])[0] // 2
            posPred = tf.slice(preds, [0], [sampNum])
            negPred = tf.slice(preds, [sampNum], [-1])
            self.preLoss += tf.reduce_mean(tf.nn.softplus(-(posPred - negPred)))
            if src == args.behNum - 1:
                self.targetPreds = preds
        self.regLoss = args.reg * Regularize()
        
        
        self.loss = self.preLoss + self.regLoss

        globalStep = tf.Variable(0, trainable=False)
        learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * args.sampNum
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = negSamp(temLabel[i], sampNum, args.item)
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur + temlen // 2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur + temlen // 2] = negloc
                cur += 1
        uLocs = uLocs[:cur] + uLocs[temlen // 2: temlen // 2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen // 2: temlen // 2 + cur]
        return uLocs, iLocs

    def trainEpoch(self):
        num = args.user
        sfIds = np.random.permutation(num)[:args.trnNum]
        epochLoss, epochPreLoss = [0] * 2
        num = len(sfIds)
        steps = int(np.ceil(num / args.batch))
        for i in range(steps):
            st = i * args.batch
            ed = min((i + 1) * args.batch, num)
            batIds = sfIds[st: ed]

            target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
            feed_dict = {}
            for beh in range(args.behNum):
                uLocs, iLocs = self.sampleTrainBatch(batIds, self.handler.trnMats[beh])
                feed_dict[self.uids[beh]] = uLocs
                feed_dict[self.iids[beh]] = iLocs

            res = self.sess.run(target, feed_dict=feed_dict,
                                options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            preLoss, regLoss, loss = res[1:]

            epochLoss += loss
            epochPreLoss += preLoss             
        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps
        return ret

    def sampleTestBatch(self, batIds, labelMat):
        batch = len(batIds)
        temTst = self.handler.tstInt[batIds]
        temLabel = labelMat[batIds].toarray()
        temlen = batch * 100
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        tstLocs = [None] * batch
        cur = 0
        for i in range(batch):
            posloc = temTst[i]
            negset = np.reshape(np.argwhere(temLabel[i] == 0), [-1])
            rdnNegSet = np.random.permutation(negset)[:99]
            locset = np.concatenate((rdnNegSet, np.array([posloc]))) 
            tstLocs[i] = locset
            for j in range(100):
                uLocs[cur] = batIds[i]
                iLocs[cur] = locset[j]
                cur += 1
        return uLocs, iLocs, temTst, tstLocs


    def testEpoch(self):
        epochHit, epochNdcg = [0] * 2 
        
        ids = self.handler.tstUsrs
        num = len(ids)
        tstBat = args.batch
        steps = int(np.ceil(num / tstBat))
        for i in range(steps):
            st = i * tstBat
            ed = min((i + 1) * tstBat, num)
            batIds = ids[st: ed]
            feed_dict = {}
            uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds, self.handler.trnMats[-1])
            feed_dict[self.uids[-1]] = uLocs
            feed_dict[self.iids[-1]] = iLocs
            preds = self.sess.run(self.targetPreds, feed_dict=feed_dict,
                                  options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            hit, ndcg = self.calcRes(np.reshape(preds, [ed - st, 100]), temTst, tstLocs)
            epochHit += hit
            epochNdcg += ndcg
        
        ret = dict()
        ret['HR'] = epochHit / num
        ret['NDCG'] = epochNdcg / num
        return ret

    def calcRes(self, preds, temTst, tstLocs):
        hit = 0
        ndcg = 0
        for j in range(preds.shape[0]):
            predvals = list(zip(preds[j], tstLocs[j]))
            predvals.sort(key=lambda x: x[0], reverse=True)
            shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
            if temTst[j] in shoot:
                hit += 1
                ndcg += np.reciprocal(np.log2(shoot.index(temTst[j]) + 2))
        return hit, ndcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        saver = tf.train.Saver()
        saver.save(self.sess, 'Models/' + args.save_path)
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        saver = tf.train.Saver()
        saver.restore(sess, 'Models/' + args.load_model)
        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    log_dir = 'log/' + args.data + '/' + os.path.basename(__file__)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    log_file = open(log_dir + '/log' + str(datetime.datetime.now()), 'w')

    def my_hook_out(text):
        log_file.write(text)
        log_file.flush()
        return 1, 0, text

    ph_out = PrintHook()
    ph_out.Start(my_hook_out)

    print("Use gpu id:", args.gpu_id)
    for arg in vars(args):
        print(arg + '=' + str(getattr(args, arg)))

    logger.saveDefault = True
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    with tf.Session(config=config) as sess:
        recom = Recommender(sess, handler)
        recom.run()