import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params_samp import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from DataHandler_samp import negSamp, transpose, DataHandler, transToLsts
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

        mets = ['Loss', 'preLoss', 'HR', 'NDCG']
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
        alluEmbed = NNs.defineParam('uEmbed0', [args.user, args.latdim // 2], reg=True)
        alliEmbed = NNs.defineParam('iEmbed0', [args.item, args.latdim // 2], reg=True)
        uEmbed0 = tf.nn.embedding_lookup(alluEmbed, self.all_usrs)
        iEmbed0 = tf.nn.embedding_lookup(alliEmbed, self.all_itms)
        allEmbed = tf.concat([uEmbed0, iEmbed0], axis = 0)

        self.ulat = [0] * (args.behNum)
        self.ilat = [0] * (args.behNum)
        for beh in range(args.behNum):
            ego_embeddings = allEmbed
            all_embeddings = [ego_embeddings]
            if args.multi_graph == False:
                for index in range(args.gnn_layer):
                    norm_embeddings = tf.sparse_tensor_dense_matmul(self.adjs[beh], all_embeddings[-1])
                    if args.encoder == 'lightgcn':
                        lightgcn_embeddings = norm_embeddings
                        all_embeddings.append(lightgcn_embeddings)
                    elif args.encoder == 'gccf':
                        gccf_embeddings = Activate(norm_embeddings, self.actFunc)
                        all_embeddings.append(gccf_embeddings)
                    elif args.encoder == 'gcn':
                        gcn_embeddings = Activate(
                            tf.matmul(norm_embeddings, self.weights['W_gc_%d' % index]) + self.weights[
                                'b_gc_%d' % index], self.actFunc)
                        all_embeddings.append(gcn_embeddings)
                    elif args.encoder == 'ngcf':
                        gcn_embeddings = Activate(
                            tf.matmul(norm_embeddings, self.weights['W_gc_%d' % index]) + self.weights[
                                'b_gc_%d' % index], self.actFunc)
                        bi_embeddings = tf.multiply(ego_embeddings, gcn_embeddings)
                        bi_embeddings = Activate(
                            tf.matmul(bi_embeddings, self.weights['W_bi_%d' % index]) + self.weights['b_bi_%d' % index],
                            self.actFunc)
                        all_embeddings.append(gcn_embeddings + bi_embeddings)

            elif args.multi_graph == True:
                for index in range(args.gnn_layer):
                    if index == 0:
                        norm_embeddings = tf.sparse_tensor_dense_matmul(self.adjs[beh], all_embeddings[-1])
                        if args.encoder == 'lightgcn':
                            lightgcn_embeddings = norm_embeddings
                            all_embeddings.append(lightgcn_embeddings + all_embeddings[-1])
                        elif args.encoder == 'gccf':
                            gccf_embeddings = Activate(norm_embeddings, self.actFunc)
                            all_embeddings.append(gccf_embeddings + all_embeddings[-1])
                        elif args.encoder == 'gcn':
                            gcn_embeddings = Activate(tf.matmul(norm_embeddings, self.weights['W_gc_%d' % index]) + self.weights['b_gc_%d' % index], self.actFunc)
                            all_embeddings.append(gcn_embeddings + all_embeddings[-1])
                        elif args.encoder == 'ngcf':
                            gcn_embeddings = Activate(tf.matmul(norm_embeddings, self.weights['W_gc_%d' % index]) + self.weights['b_gc_%d' % index], self.actFunc)
                            bi_embeddings = tf.multiply(ego_embeddings, gcn_embeddings)
                            bi_embeddings = Activate(tf.matmul(bi_embeddings, self.weights['W_bi_%d' % index]) + self.weights['b_bi_%d' % index], self.actFunc)
                            all_embeddings.append(gcn_embeddings + bi_embeddings + all_embeddings[-1])
                    else:
                        atten = FC(ego_embeddings, args.behNum, reg=True, useBias=True,
                                   activation=self.actFunc, name='attention_%d_%d'%(beh,index), reuse=True)
                        temp_embeddings = []
                        for inner_beh in range(args.behNum):
                            neighbor_embeddings = tf.sparse_tensor_dense_matmul(self.adjs[inner_beh], norm_embeddings)
                            temp_embeddings.append(neighbor_embeddings)
                        all_temp_embeddings = tf.stack(temp_embeddings, 1)
                        norm_embeddings = tf.reduce_sum(tf.einsum('abc,ab->abc', all_temp_embeddings, atten), axis=1, keepdims=False)
                        if args.encoder == 'lightgcn':
                            lightgcn_embeddings = norm_embeddings
                            all_embeddings.append(lightgcn_embeddings + all_embeddings[-1])
                        elif args.encoder == 'gccf':
                            gccf_embeddings = Activate(norm_embeddings, self.actFunc)
                            all_embeddings.append(gccf_embeddings + all_embeddings[-1])
                        elif args.encoder == 'gcn':
                            gcn_embeddings = Activate(tf.matmul(norm_embeddings, self.weights['W_gc_%d' % index]) + self.weights['b_gc_%d' % index], self.actFunc)
                            all_embeddings.append(gcn_embeddings + all_embeddings[-1])
                        elif args.encoder == 'ngcf':
                            gcn_embeddings = Activate(tf.matmul(norm_embeddings, self.weights['W_gc_%d' % index]) + self.weights['b_gc_%d' % index], self.actFunc)
                            bi_embeddings = tf.multiply(ego_embeddings, gcn_embeddings)
                            bi_embeddings = Activate(tf.matmul(bi_embeddings, self.weights['W_bi_%d' % index]) + self.weights['b_bi_%d' % index], self.actFunc)
                            all_embeddings.append(gcn_embeddings + bi_embeddings + all_embeddings[-1])

            all_embeddings = tf.add_n(all_embeddings)
            self.ulat[beh], self.ilat[beh] = tf.split(all_embeddings, [tf.shape(self.all_usrs)[0], tf.shape(self.all_itms)[0]], 0)
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
        if args.normalization == "left":
            norm_adj_mat = left_adj_single(adj_mat)
        elif args.normalization == "right":
            norm_adj_mat = right_adj_single(adj_mat)
        elif args.normalization == "symm":
            norm_adj_mat = symm_adj_single(adj_mat) 
        elif args.normalization == 'none':
            norm_adj_mat = adj_mat            
        return norm_adj_mat.tocsr()

    def prepareModel(self):
        self.actFunc = 'leakyRelu'
        self.adjs = []
        self.uids, self.iids = [], []
        for i in range(args.behNum):
            self.adjs.append(tf.sparse_placeholder(dtype=tf.float32))
            self.uids.append(tf.placeholder(name='uids' + str(i), dtype=tf.int32, shape=[None]))
            self.iids.append(tf.placeholder(name='iids' + str(i), dtype=tf.int32, shape=[None]))
        self.all_usrs = tf.placeholder(name='all_usrs', dtype=tf.int32, shape=[None])
        self.all_itms = tf.placeholder(name='all_itms', dtype=tf.int32, shape=[None])

        self.defineModel() 

        self.preLoss = 0
        for src in range(args.behNum):
            if args.decoder == 'bilinear':
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

    def sampleTrainBatch(self, batIds, labelMat,itmNum):
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
                neglocs = negSamp(temLabel[i], sampNum, itmNum)
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
        allIds = np.random.permutation(num)[:args.trnNum]
        epochLoss, epochPreLoss = [0] * 2
        glbnum = len(allIds)
        glb_step = int(np.ceil(glbnum / args.batch))
       
        bigSteps = int(np.ceil(glbnum / args.divSize))      
        
        for s in range(bigSteps):
            bigSt = s * args.divSize
            bigEd = min((s+1) * args.divSize, glbnum)
            sfIds = allIds[bigSt: bigEd]
            num = bigEd - bigSt

            steps = num // args.batch

            pckAdjs, pckTpAdjs, usrs, itms = self.handler.sampleLargeGraph(sfIds)
            usrIdMap = dict(map(lambda x: (usrs[x], x), range(len(usrs))))
            sfIds = list(map(lambda x: usrIdMap[x], sfIds))
            feed_dict = {self.all_usrs: usrs, self.all_itms: itms}

            for i in range(args.behNum):		
                coomat_r = sp.coo_matrix(pckAdjs[i])
                coomat_rt = sp.coo_matrix(pckTpAdjs[i])     
                row = np.concatenate([coomat_r.row,coomat_rt.row+pckAdjs[i].shape[0]])
                col = np.concatenate([coomat_r.col+pckAdjs[i].shape[0],coomat_rt.col])
                data = np.concatenate([coomat_r.data.astype(np.float32),coomat_rt.data.astype(np.float32)])
                adj_mat = sp.coo_matrix((data, (row, col)), shape=(coomat_r.shape[0]+coomat_r.shape[1], coomat_r.shape[0]+coomat_r.shape[1]))
                norm_trn = self.create_multiple_adj_mat(adj_mat)
                idx, data, shape = transToLsts(norm_trn, norm=False)
                feed_dict[self.adjs[i]] = idx, data, shape  
                
            for i in range(steps):
                st = i * args.batch
                ed = min((i + 1) * args.batch, num)
                batIds = sfIds[st: ed]

                target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
                
                for beh in range(args.behNum):
                    uLocs, iLocs = self.sampleTrainBatch(batIds, pckAdjs[beh],len(itms))
                    feed_dict[self.uids[beh]] = uLocs
                    feed_dict[self.iids[beh]] = iLocs

                res = self.sess.run(target, feed_dict=feed_dict,
                                    options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
                preLoss, regLoss, loss = res[1:]

                epochLoss += loss
                epochPreLoss += preLoss
        ret = dict()
        ret['Loss'] = epochLoss / glb_step
        ret['preLoss'] = epochPreLoss / glb_step
        return ret

    def sampleTestBatch(self, batIds, labelMat, tstInt):
        batch = len(batIds)
        temTst = tstInt[batIds]
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
        allIds = self.handler.tstUsrs
        glbnum = len(allIds)
        tstBat = args.batch
        bigSteps = int(np.ceil(glbnum / args.divSize))
        glb_step = int(np.ceil(glbnum / tstBat))
        for s in range(bigSteps):
            bigSt = s * args.divSize
            bigEd = min((s+1) * args.divSize, glbnum)
            ids = allIds[bigSt: bigEd]
            num = bigEd - bigSt

            steps = int(np.ceil(num / tstBat))

            posItms = self.handler.tstInt[ids]
            pckAdjs, pckTpAdjs, usrs, itms = self.handler.sampleLargeGraph(ids, list(set(posItms)))
            usrIdMap = dict(map(lambda x: (usrs[x], x), range(len(usrs))))
            itmIdMap = dict(map(lambda x: (itms[x], x), range(len(itms))))
            ids = list(map(lambda x: usrIdMap[x], ids))
            itmMapping = (lambda x: None if (x is None or x not in itmIdMap) else itmIdMap[x])
            pckTstInt = np.array(list(map(lambda x: itmMapping(self.handler.tstInt[usrs[x]]), range(len(usrs)))))
            feed_dict = {self.all_usrs: usrs, self.all_itms: itms}
            for i in range(args.behNum):
                coomat_r = sp.coo_matrix(pckAdjs[i])
                coomat_rt = sp.coo_matrix(pckTpAdjs[i])     
                row = np.concatenate([coomat_r.row,coomat_rt.row+pckAdjs[i].shape[0]])
                col = np.concatenate([coomat_r.col+pckAdjs[i].shape[0],coomat_rt.col])
                data = np.concatenate([coomat_r.data.astype(np.float32),coomat_rt.data.astype(np.float32)])
                adj_mat = sp.coo_matrix((data, (row, col)), shape=(coomat_r.shape[0]+coomat_r.shape[1], coomat_r.shape[0]+coomat_r.shape[1]))
                norm_trn = self.create_multiple_adj_mat(adj_mat)
                idx, data, shape = transToLsts(norm_trn, norm=False)
                feed_dict[self.adjs[i]] = idx, data, shape                 
        
            for i in range(steps):
                st = i * tstBat
                ed = min((i + 1) * tstBat, num)
                batIds = ids[st: ed]
                uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds, pckAdjs[-1],pckTstInt)
                feed_dict[self.uids[-1]] = uLocs
                feed_dict[self.iids[-1]] = iLocs
                preds = self.sess.run(self.targetPreds, feed_dict=feed_dict,
                                    options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
                hit, ndcg = self.calcRes(np.reshape(preds, [ed - st, 100]), temTst, tstLocs)
                epochHit += hit
                epochNdcg += ndcg
        ret = dict()
        ret['HR'] = epochHit / glbnum
        ret['NDCG'] = epochNdcg / glbnum
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