"""Generate Grad-CAM++ explanations for the EfficientNetB0 application model."""

import numpy as np


def gradcam_plus_plus_heatmap(tf, gradcam_model, image_pixels):
    """Compute normalized Grad-CAM++ using gradient-power alpha weights."""
    inputs = tf.convert_to_tensor(
        image_pixels[np.newaxis, ...],
        dtype=tf.float32,
    )
    with tf.GradientTape() as tape:
        activations, probabilities = gradcam_model(inputs, training=False)
        predicted_index = int(tf.argmax(probabilities[0]).numpy())
        class_score = probabilities[:, predicted_index]

    gradients = tape.gradient(class_score, activations)
    if gradients is None:
        raise ValueError("Grad-CAM++ gradients are unavailable.")
    if not bool(tf.reduce_all(tf.math.is_finite(gradients)).numpy()):
        raise ValueError("Grad-CAM++ gradients contain NaN or infinite values.")

    feature_maps = activations[0]
    first_gradients = gradients[0]
    squared_gradients = tf.square(first_gradients)
    cubed_gradients = squared_gradients * first_gradients
    third_order_term = tf.reduce_sum(
        feature_maps * cubed_gradients,
        axis=(0, 1),
    )
    denominator = (
        2.0 * squared_gradients
        + third_order_term[tf.newaxis, tf.newaxis, :]
    )
    alphas = squared_gradients / tf.maximum(
        denominator,
        tf.constant(1e-8, dtype=denominator.dtype),
    )
    alphas = tf.where(
        tf.math.is_finite(alphas),
        alphas,
        tf.zeros_like(alphas),
    )
    channel_weights = tf.reduce_sum(
        alphas * tf.nn.relu(first_gradients),
        axis=(0, 1),
    )
    heatmap = tf.nn.relu(
        tf.reduce_sum(
            feature_maps * channel_weights[tf.newaxis, tf.newaxis, :],
            axis=-1,
        )
    )
    maximum = float(tf.reduce_max(heatmap).numpy())
    if not np.isfinite(maximum) or maximum <= 1e-12:
        raise ValueError("Grad-CAM++ produced an empty heatmap.")
    heatmap_array = np.asarray(heatmap.numpy() / maximum, dtype=np.float32)
    if heatmap_array.ndim != 2 or not np.all(np.isfinite(heatmap_array)):
        raise ValueError("Grad-CAM++ produced an invalid heatmap.")
    return (
        heatmap_array,
        predicted_index,
        float(probabilities[0, predicted_index].numpy()),
    )
