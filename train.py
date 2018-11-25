# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:17:08 2018

@author: hty
"""
'''
原文vsdr采用x2，x3，x4三种数据一起训练，这里只采用了x2
'''
import tensorflow as tf
import sys
import time
from utility import *
import shutil
import os


        
def vdsr_train(train_data_file,test_data_file,model_save_path):
    train_data, train_label = read_data(train_data_file)
    test_data, test_label = read_data(test_data_file)
    
    batch_size     = 64
    iterations     = train_data.shape[0]//batch_size
    total_epoch    = 80#这里是指还需要继续训练多少个epoch
    lr             = 0.0001
    image_size     = 41
    label_size     = 41
    
    is_load = False#是否加载现有模型进行再训练
    per_epoch_save = 1
    start_epoch = 0
    images = tf.placeholder(tf.float32, [None, image_size, image_size, 1], name='images')
    labels = tf.placeholder(tf.float32, [None, label_size, label_size, 1], name='labels')
    learning_rate = tf.placeholder(tf.float32)
    
    pred = vdsr(images)
    loss = tf.reduce_mean(tf.square(labels - pred))
    #l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    psnr = PSNR(labels,pred)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate,use_nesterov=True).minimize(loss + l2 * weight_decay)
    saver = tf.train.Saver()
    
    
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())     
        if is_load==True:
            start_epoch = 48
            check_point_path = model_save_path + '/' + str(start_epoch) +'/' # 保存好模型的文件路径
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        bar_length = 25
        for ep in range(1+start_epoch,total_epoch+1+start_epoch):
            start_time = time.time()
            pre_index = 0
            train_loss = 0.0
            train_psnr = 0.0
            print("\nepoch %d/%d, lr = %2.5f:" %(ep,start_epoch+total_epoch,lr))
            indices = np.random.permutation(len(train_data))#每次随机打乱数据
            train_data = train_data[indices]
            train_label = train_label[indices]
            for it in range(1,iterations+1):
                batch_x = train_data[pre_index:pre_index+batch_size]
                batch_y = train_label[pre_index:pre_index+batch_size]
                _, batch_loss, batch_psnr = sess.run([train_step, loss,psnr],feed_dict={images:batch_x, labels:batch_y, learning_rate: lr})
                
                train_loss += batch_loss
                train_psnr += batch_psnr
                pre_index  += batch_size
                
                if it == iterations:
                    train_loss /= iterations
                    train_psnr /= iterations
                    test_loss, test_psnr  = sess.run([loss,psnr],feed_dict={images:test_data, labels:test_label})
                    
                    s1 = "\r%d/%d [%s%s] - batch_time = %.2fs - train_loss = %.5f - train_psnr = %.2f"%(it,iterations,">"*(bar_length*it//iterations),"-"*(bar_length-bar_length*it//iterations), (time.time()-start_time)/it, train_loss, train_psnr)#run_test()
                    sys.stdout.write(s1)
                    sys.stdout.flush()
                    
                    print("\ncost_time: %ds, test_loss: %.5f, test_psnr: %.2f" %(int(time.time()-start_time), test_loss, test_psnr))
                    '''
                    这里输出的test_psnr并不是最终Set5的psnr，而是图像块的平均值
                    '''
                else:
                    s1 = "\r%d/%d [%s%s] - batch_time = %.2fs - train_loss = %.5f - train_psnr = %.2f"%(it,iterations,">"*(bar_length*it//iterations),"-"*(bar_length-bar_length*it//iterations), (time.time()-start_time)/it, train_loss / it, train_psnr / it)#run_test()
                    sys.stdout.write(s1)
                    sys.stdout.flush()
            if ep % per_epoch_save ==0:
                path = model_save_path + '/save/' + str(ep) + '/'
                save_model = saver.save(sess, path+'vdsr_model')
                new_path = model_save_path + '/' + str(ep) + '/'
                shutil.move(path,new_path)
                '''
                模型首先是被保存在save下面的,直接保存的话，前面的epoch对应的文件夹会出现内部文件被删除的情况，原因不明；所以这里用shutil.move把模型所在的文件夹移动了一下
                '''
                print("\nModel saved in file: %s" % save_model)
        path = './final_model/vdsr_model'
        save_model = saver.save(sess, path)
        print("\nModel saved in file: %s" % save_model)

    
if __name__ == '__main__':
    
    train_file = 'train.h5'
    test_file = 'test.h5'
    model_save_path = 'vdsr_checkpoint'
    
    if os.path.exists(model_save_path)==False:
        print('The ' + '"' +model_save_path + '"'+ 'can not find! Create now!')
        os.mkdir(model_save_path)
        
    if os.path.exists(model_save_path+'\save')==False:
        print('The ' + '"save' + '"' +' can not find! Create now!')
        os.mkdir(model_save_path+'\save')
        
    vdsr_train(train_file,test_file,model_save_path)