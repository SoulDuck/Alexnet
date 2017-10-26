import tensorflow as tf

def aug_lv0(image_ , is_training , image_size):


    def aug_with_train(image, image_size):
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image = tf.random_crop(image, [image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        # Brightness / saturatio / constrast provides samll gains 2%~5% on cifar

        image = tf.image.random_brightness(image, max_delta=63. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.8)
        image = tf.image.per_image_standardization(image)
        image=tf.identity(image ,name='image_augmented')
        return image

    def aug_with_test(image , image_size):
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image = tf.image.per_image_standardization(image)
        return image

    image=tf.cond(is_training , lambda : aug_with_train(image_ , image_size=image_size)  , \
                  lambda image : aug_with_test(image_ , image_size=image_size) ,)
    return image

