from __future__ import print_function
import tensorflow as tf
import numpy as np
import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import time
import BatchDatsetReader as dataset
from six.moves import xrange
import os
import metrics_output
from Inference2 import inference
from evaluation_object import cal_loss, normal_loss, per_class_acc, get_hist, print_hist_summary, train_op, MAX_VOTE, var_calculate
import gc
os.environ["CUDA_VISIBLE_DEVICES"]="2"
logdir="../logs/resunet_pool2_STRONG/"
data_dir = "../Data/DATA_STRONG/"

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", logdir, "path to logs directory")
tf.flags.DEFINE_string("data_dir", data_dir, "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-6", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MAX_ITERATION = int(1e5 + 5e4+ 1)
NUM_OF_CLASSESS = 3
IMAGE_SIZE = 1024
size1=1
size=size1*4

def train(loss_val, var_list):
    print('we are computing the loss grads now !!!')
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)

def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE/size1, IMAGE_SIZE/size1, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE/size, IMAGE_SIZE/size, 1], name="annotation")
    pred_annotation,logits= inference(image,'pool2', keep_probability,isDeconv=True)
    #cross_entropy_mean, accuracy, _=normal_loss(pred_annotation, annotation, NUM_OF_CLASSESS)

    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    '''
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    '''
    '''
    loss = -utils.IOU_(pred_annotation, annotation)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = utils.make_train_op(pred_annotation, annotation)

    IOU_op = IOU_(pred_annotation, annotation)
    IOU_op = tf.Print(IOU_op, [IOU_op])
    '''
    #loss=utils.focal_loss_softmax(labels=annotation,logits=logits)
    loss = utils.focal_loss(prediction_tensor = logits, target_tensor = annotation)
    loss_summary = tf.summary.scalar("focal loss", loss)

    trainable_var = tf.trainable_variables()

    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)

    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE/size1,'resize_anno': True, 'resize_size_anno': IMAGE_SIZE/size}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    print("Setting up Saver...")
    #saver = tf.train.Saver(max_to_keep=10)
    saver = tf.train.Saver(max_to_keep=50)

    # create two summary writers to show training loss and validation loss in the same graph
    # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    ###############################################################################################
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    ###############################################################################################

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            gc.collect()
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)

            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}
            train_start = time.clock()
            sess.run(train_op, feed_dict=feed_dict)
            train_elapsed = (time.clock() - train_start)

            if itr % 10 == 0:

                train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:

                #output
                print('Step:',itr)
                path=FLAGS.logs_dir+'visualize/step_'+str(itr)
                folder = os.path.exists(path)
                if folder:
                    print('Aiready exists:',path)
                else:
                    os.makedirs(path)
                valid_loss_all=0
                for num in range(len(valid_records)):
                    valid_images, valid_annotations = validation_dataset_reader.next_batch(1)

                    vali_start = time.clock()
                    pred,valid_loss = sess.run([pred_annotation,loss], feed_dict={image: valid_images, annotation: valid_annotations,keep_probability: 1.0})
                    vali_elapsed = (time.clock() - vali_start)

                    valid_loss_all=valid_loss_all+valid_loss
                    valid_annotations = np.squeeze(valid_annotations, axis=3)
                    pred = np.squeeze(pred, axis=3)
                    for itr_ in range(1):
                        utils.save_image(valid_annotations[itr_].astype(np.uint8), path, name="gt_" +str(num)+'_'+str(itr_))
                        utils.save_image(pred[itr_].astype(np.uint8), path, name="pred_" + str(num)+'_'+str(itr_))
                print('save image:',len(valid_records))
                valid_loss_mean=valid_loss_all/len(valid_records)
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss_mean))
                # add validation loss to TensorBoard
                #summary_sva = tf.summary.scalar("vali focal loss", valid_loss_mean)
                #validation_writer.add_summary(sess.run(summary_sva), itr)

            if itr % 2000==0:
                time_file=open(r'time.txt','a')
                time_file.write(str(itr)+': '+str(train_elapsed)+' '+str(vali_elapsed)+'\n')
                time_file.close()
                print("train_Inference_time:",train_elapsed)
                print("vali_Inference_time:",vali_elapsed)

            if itr % 10000==0:
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    '''
    elif FLAGS.mode == "test":
        for step_id in range(len(MODEL_STEP_Group)):
            MODEL_STEP=MODEL_STEP_Group[step_id]
            print('MODEL_STEP:',MODEL_STEP)
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.import_meta_graph(FLAGS.logs_dir+'model.ckpt-'+str(MODEL_STEP)+'.meta')
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored...")

            path=FLAGS.logs_dir+'test/step_'+str(MODEL_STEP)+'/'
            folder = os.path.exists(path)
            if folder:                 
                print('Aiready exists:',path)
            else:
                os.makedirs(path)
            Inference_time=0
            Read_Img_Time=0
            for num in range(len(valid_records)/FLAGS.batch_size):
                ##########################################
                Read_Img_Start=time.clock()
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                Read_Img_elapsed=time.clock()-Read_Img_Start
                #print("Read_Img_Time:",Read_Img_elapsed)
                Read_Img_Time=Read_Img_Time+Read_Img_elapsed

                feed_dict={image: valid_images, annotation: valid_annotations,keep_probability: 1.0}

                Inference_start = time.clock()
                pred ,logits_,logits_softmax_= sess.run([pred_annotation,logits,logits_softmax], feed_dict=feed_dict)
                Inference_elapsed = (time.clock() - Inference_start)
                Inference_time=Inference_time+Inference_elapsed
                #print("InferenceTime:",Inference_elapsed)
                ###########################################
                valid_annotations = np.squeeze(valid_annotations, axis=3)
                pred = np.squeeze(pred, axis=3)

                for itr in range(FLAGS.batch_size):
                    idNUM=str(FLAGS.batch_size*num+itr)
                    
                    utils.save_image(valid_images[itr].astype(np.uint8), path, name="inp_" + idNUM)
                    utils.save_image(valid_annotations[itr].astype(np.uint8), path, name="gt_" +idNUM)
                    utils.save_image(pred[itr].astype(np.uint8), path, name="pred_" + idNUM)
                    #********************************************************
                    logits_2=logits_[itr,:,:,:]
                    logits_softmax_2=logits_softmax_[itr,:,:,:]
                    utils.save_image(logits_2.astype(np.uint8),path, name="heatmap_"  + idNUM)
                    utils.save_image(logits_softmax_2[:,:,0],  path, name="heatmap_bg_" + idNUM)
                    utils.save_image(logits_softmax_2[:,:,1],  path, name="heatmap_ab_" + idNUM)
                    utils.save_image(logits_softmax_2[:,:,2],  path, name="heatmap_no_" + idNUM)
                    #********************************************************
                    
                    #print("Saved image: %d" % (FLAGS.batch_size*num+itr))

            print("MeanInferenceTime:",Inference_time/len(valid_records))
            print("MeanRead_Img_Time:",Read_Img_Time/len(valid_records))

    elif FLAGS.mode == "uncertain_test":
        start = time.clock()

        IsBayes=True
        num_sample_generate = 15
        acc_final = []
        iu_final = []
        iu_mean_final = []

        loss_tot = []
        acc_tot  = []
        pred_tot = []
        var_tot  = []
        hist = np.zeros((NUM_OF_CLASSESS, NUM_OF_CLASSESS))
        step = 0
        for num in range(len(valid_records)):
            valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)

            # comment the code below to apply the dropout for all the samples
            if num_sample_generate == 1:
                feed_dict={image: valid_images, annotation: valid_annotations,keep_probability: 1.0}
            else:
                feed_dict={image: valid_images, annotation: valid_annotations,keep_probability: 0.5}
            # uncomment this code below to run the dropout for all the samples
            # feed_dict = {test_data_tensor: image_batch, test_label_tensor:label_batch, phase_train: False, keep_prob:0.5, phase_train_dropout:True}
            #fetches = [loss, accuracy, self.logits, prediction]
            if IsBayes is False:
                #loss_per, acc_per, logit, pred = sess.run(fetches=fetches, feed_dict=feed_dict)
                var_one = []
            else:
                logit_iter_tot = []
                loss_iter_tot = []
                acc_iter_tot = []
                prob_iter_tot = []
                prob_iter_tot0 = []
                prob_iter_tot1 = []
                prob_iter_tot2 = []
                logit_iter_temp = []
                for iter_step in range(num_sample_generate):
                    pred,logit_iter_step,prob_iter_step,loss_iter_step, logits_softmax2,acc_iter_step= sess.run([pred_annotation, logits,prob,loss,logits_softmax,accuracy],feed_dict=feed_dict)

                    #valid_annotations = np.squeeze(valid_annotations, axis=3)
                    pred = np.squeeze(pred, axis=3)
                    loss_iter_tot.append(loss_iter_step)
                    acc_iter_tot.append(acc_iter_step)
                    logit_iter_tot.append(logit_iter_step)

                    prob_iter_tot.append(prob_iter_step)
                    prob_iter_tot0.append(prob_iter_step[:,:,0])
                    prob_iter_tot1.append(prob_iter_step[:,:,1])
                    prob_iter_tot2.append(prob_iter_step[:,:,2])



                    logit_iter_temp.append(
                        np.reshape(logit_iter_step, [IMAGE_SIZE/size, IMAGE_SIZE/size, NUM_OF_CLASSESS]))

                loss_per = np.nanmean(loss_iter_tot)
                acc_per = np.nanmean(acc_iter_tot)
                logit = np.nanmean(logit_iter_tot, axis=0)
                print(np.shape(prob_iter_tot))

                prob_mean = np.nanmean(prob_iter_tot, axis=0)
                prob_variance  = np.var(prob_iter_tot , axis=0)
                #prob_variance0 = np.var(prob_iter_tot0, axis=0)
                #prob_variance1 = np.var(prob_iter_tot1, axis=0)
                #prob_variance2 = np.var(prob_iter_tot2, axis=0)
                logit_variance = np.var(logit_iter_temp, axis=0)

                # THIS TIME I DIDN'T INCLUDE TAU
                pred2 = np.reshape(np.argmax(prob_mean, axis=-1), [-1])  # pred is the predicted label

                var_sep = []  # var_sep is the corresponding variance if this pixel choose label k

                prob_variance_all=np.reshape(prob_variance, [(IMAGE_SIZE/size) * (IMAGE_SIZE/size), NUM_OF_CLASSESS])
                length_cur = 0  # length_cur represent how many pixels has been read for one images
                for row in prob_variance_all:
                    temp = row[pred2[length_cur]]

                    length_cur += 1
                    var_sep.append(temp)
                var_one = np.reshape(var_sep, [IMAGE_SIZE/size,IMAGE_SIZE/size])  # var_one is the corresponding variance in terms of the "optimal" label

                pred2 = np.reshape(pred2, [IMAGE_SIZE/size, IMAGE_SIZE/size])

            loss_tot.append(loss_per)
            acc_tot.append(acc_per)
            pred_tot.append(pred2)
            var_tot.append(var_one)

            print("Image Index {}: TEST Loss{:6.3f}, TEST Accu {:6.3f}".format(step, loss_tot[-1], acc_tot[-1]))

            path=FLAGS.logs_dir+'visualize/test'
            print('path:',path)
            folder = os.path.exists(path)
            if folder:
                print('Aiready exists:',path)
            else:
                os.makedirs(path)
            
            itr=0
            labels_image_array = np.squeeze(valid_annotations, axis=3)
            #utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir+'visualize/test', name="inp_" + str(num)+'_'+str(itr))
            #utils.save_image(labels_image_array[itr].astype(np.uint8), FLAGS.logs_dir+'visualize/test', name="gt_" +str(num)+'_'+str(itr))
            #utils.save_image(pred2.astype(np.uint8), FLAGS.logs_dir+'visualize/test', name="pred_" + str(num)+'_'+str(itr))
            #utils.save_image(var_one, FLAGS.logs_dir+'visualize/test', name="uncertainty_" + str(num)+'_'+str(itr))
            #utils.save_image(prob_variance[:,:,0], FLAGS.logs_dir+'visualize/test', name="uncertainty0_" + str(num)+'_'+str(itr))
            #utils.save_image(prob_variance[:,:,1], FLAGS.logs_dir+'visualize/test', name="uncertainty1_" + str(num)+'_'+str(itr))
            #utils.save_image(prob_variance[:,:,2], FLAGS.logs_dir+'visualize/test', name="uncertainty2_" + str(num)+'_'+str(itr))

            step = step + 1
            per_class_acc(logit, valid_annotations, NUM_OF_CLASSESS)
            hist += get_hist(logit, valid_annotations)

            print('********************************************************')

        acc_tot = np.diag(hist).sum() / hist.sum()
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

        print("Total Accuracy for test image: ", acc_tot)
        print("Total MoI for test images: ", iu)
        print("mean MoI for test images: ", np.nanmean(iu))
        acc_final.append(acc_tot)
        iu_final.append(iu)
        iu_mean_final.append(np.nanmean(iu))

        elapsed = (time.clock() - start)
        print("Time used2:",elapsed)

    #return acc_final, iu_final, iu_mean_final, prob_variance, logit_variance, pred_tot, var_tot
    '''

if __name__ == "__main__":
    #tf.app.run()
    import cv2
    data_path=r'/home/xing/Desktop/smoke_video_data(under_channel)'
    img_data_path=r'/home/xing/Desktop/smoke_img_data(under_channel)'
    for video_name in os.listdir(data_path):
        print(video_name)
        video_path=os.path.join(data_path,video_name)
        video_img_path = os.path.join(img_data_path, video_name)
        video_capture=cv2.VideoCapture(video_path)
        frame_num=0
        success=True
        if not os.path.exists(video_img_path):
            os.mkdir(video_img_path)

        while(success):
            success,frame=video_capture.read()
            if not success:
                break
            frame_num+=1
            cv2.imwrite(os.path.join(video_img_path,
            video_name+'_'+str(frame_num)+'_0_0_0_0_1.jpg'),
                        frame)
