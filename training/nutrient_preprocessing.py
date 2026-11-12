"""Shared inference and training image preprocessing for nutrient EfficientNetB0."""


def preprocess_image_path(image_path):
    """Decode RGB and resize to 224x224 float pixels in the [0, 255] range.

    EfficientNetB0 includes its own rescaling layer, so pixels must not be
    divided by 255 or normalized a second time.
    """
    import tensorflow as tf

    encoded = tf.io.read_file(image_path)
    image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(
        image,
        (224, 224),
        method="bilinear",
        antialias=True,
    )
    return tf.cast(image, tf.float32)
