import tensorflow as tf

with tf.Session() as sess:
    saver = tf.train.Saver()

    # Restore variables from disk.
    saver.restore(sess, "model/3_Step-18333.ckpt")
    print("Model restored.")
