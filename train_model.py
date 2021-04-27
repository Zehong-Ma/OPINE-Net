'''
@File    :   train_model.py
@Author  :   Zehong Ma
@Version :   1.0
@Contact :   zehongma@qq.com
@Desc    :   define the tensorflow graph and optimizer, and train the OPINE-Net(plus)
'''

import tensorflow as tf
import scipy.io as sio
from deployment import model_deploy
from modules import *
from utils import parse_record
from configuration import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS



def _average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            # print(g)
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train():

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        #######################
        # Config model_deploy #
        #######################
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)

        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        ######################
        # load the data #
        ######################
        dataset = tf.data.TFRecordDataset(FLAGS.data_dir+'/'+'Training_Data.tfrecord')
        dataset = dataset.map(parse_record)
        dataset = dataset.repeat(FLAGS.epochs+1).shuffle(20000).batch(FLAGS.batch_size)
        num_batches = FLAGS.epochs*FLAGS.num_samples//FLAGS.batch_size
        iterator = dataset.make_one_shot_iterator()
        batch_x = iterator.get_next()
        batch_x = tf.reshape(batch_x,[FLAGS.batch_size,33,33,1])  
        

        ###########################
        # Model                   #
        ###########################
        
        Phix, Phi_weight, Phi = sampling_subnet(batch_x)
        x_0, Phi_T_weight = initialization_subnet(Phix, Phi)
        x_final, layers_sym = recovery_subnet(x_0, Phi_weight, Phi_T_weight)


        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = configure_learning_rate(FLAGS.num_samples, global_step)
            optimizer = configure_optimizer(learning_rate)

        tower_grads = []
        tower_grads_without_imageNet = []
        image_fc_grads=[]
        for k in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % k):
                with tf.name_scope('tower_%d' % k) as scope:
                    with tf.variable_scope(tf.get_variable_scope()):

                        # loss, cmpm_loss, cmpc_loss, i2t_loss, t2i_loss, image_fc_similarity_loss, text_fc_similarity_loss= \
                        #     _tower_loss(network_fn, images_splits[k], labels_splits[k],
                        #                 input_seqs_splits[k], input_masks_splits[k])

                        #############################################################
                        #                         loss compute                      #
                        #############################################################               

                        # loss_discrepancy
                        loss_discrepancy = tf.reduce_mean(tf.pow((x_final - batch_x), 2))
                        
                        # loss_symmetry
                        n_input = ratio_dict[FLAGS.cs_ratio]
                        Eye_I = tf.eye(n_input)
                        loss_symmetry = tf.reduce_mean(tf.pow(layers_sym[0], 2))
                        for k in range(FLAGS.layer_num-1):
                            loss_symmetry += tf.reduce_mean(tf.pow(layers_sym[k+1], 2))

                        # loss_orth
                        loss_orth = tf.reduce_mean(tf.pow((tf.matmul(Phi, tf.transpose(Phi, [1,0]))-Eye_I), 2)  )

                        loss = loss_discrepancy + 0.01*loss_symmetry + 0.01*loss_orth
                        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                        #total_loss = tf.add_n([loss] + reg_loss, name='total_loss')
                        total_loss = loss
                        #the next code line's function is to exponential moving average when update occurs
                        #  eg: 0.9*(loss1+total_loss1)+0.1*(update_loss1+update_total_loss1)
                        # loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg_loss')
                        # loss_averages_op = loss_averages.apply([loss] + [total_loss])
                        # loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg_loss')
                        # loss_averages_op = loss_averages.apply([loss] + [total_loss])
                        # with tf.control_dependencies([loss_averages_op]):
                        #     total_loss = tf.identity(total_loss)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        summaries.append(tf.summary.scalar('Total_loss', loss))
                        summaries.append(tf.summary.scalar('loss_discrepancy', loss_discrepancy))


                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)

                        # Variables to train.
                        variables_to_train = tf.trainable_variables()
                
                        grads = optimizer.compute_gradients(total_loss, var_list=variables_to_train)

                    

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        # grads = _average_gradients(tower_grads)

        

        # Add a summary to track the learning rate and precision.
        summaries.append(tf.summary.scalar('learning_rate', learning_rate))

        # Add histograms for histogram and trainable variables.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        #################################
        # Configure the moving averages #
        #################################
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
            update_ops.append(variable_averages.apply(moving_average_variables))
            

        # Apply the gradients to adjust the shared variables.
        grad_updates = optimizer.apply_gradients(grads, global_step=global_step)
        
        update_ops.append(grad_updates)
       
        # Group all updates to into a single train op.
        train_op = tf.group(*update_ops)
        
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=None)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU implementations.
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement)

        sess = tf.Session(config=config)
        sess.run(init)

        ck_global_step = get_init_fn(sess)
        print_train_info()

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(
            os.path.join(FLAGS.log_dir),
            graph=sess.graph)

        for step in xrange(num_batches):
            step+=int(ck_global_step)
            _,total_loss_value, loss_discrepancy_value ,loss_symmetry_value ,loss_orth_value = sess.run([train_op,total_loss,loss_discrepancy,loss_symmetry,loss_orth])
          
            if step % 100 == 0:
                format_str = ('step %d, total_loss = %.4f, loss_discrepancy = %.4f ,loss_symmetry = %.4f, loss_orth = %.4f')
                print(format_str % (step, total_loss_value, loss_discrepancy_value,loss_symmetry_value,loss_orth_value))
            
            if step % 2000 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % FLAGS.ckpt_steps == 0 or (step + 1) == num_batches:
                checkpoint_path = os.path.join(FLAGS.model_dir, 'model_layer%d_group%d_ratio%d.ckpt'
                                                                    %(FLAGS.layer_num,FLAGS.group_num,FLAGS.cs_ratio))
                saver.save(sess, checkpoint_path, global_step=step)
