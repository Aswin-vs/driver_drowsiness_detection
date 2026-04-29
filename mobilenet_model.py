"""
mobilenet_model.py
------------------
Transfer learning model using MobileNetV2 pre-trained on ImageNet.

Training Strategy:

    Phase 1 -- Feature Extraction (10 epochs, LR = 1e-3):
        Freeze the entire MobileNetV2 base.
        Train only the custom classification head.
        Lets the head converge before any base weights are touched, preventing
        large gradients from the randomly-initialised head from corrupting
        the pre-trained base.

    Phase 2 -- Fine-Tuning (20 epochs, LR = 1e-5):
        Unfreeze the top FINE_TUNE_AT layers of the base.
        Train at a very low learning rate to adapt high-level ImageNet
        features to the drowsiness domain without destroying general
        low-level features.
        BatchNorm layers are kept frozen throughout to preserve their
        running statistics.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization,
)
from tensorflow.keras.optimizers import Adam

# Number of layers from the END of MobileNetV2 to unfreeze during fine-tuning.
# MobileNetV2 has 154 layers total; unfreezing 100 exposes its higher-level
# feature extractors while keeping early edge/texture detectors frozen.
FINE_TUNE_AT = 100


def build_mobilenet(input_shape=(224, 224, 3), learning_rate=1e-3):
    """
    Phase 1 model: frozen MobileNetV2 base + trainable classification head.

    Parameters
    ----------
    input_shape   : tuple  -- must be (224, 224, 3) for MobileNetV2
    learning_rate : float  -- Adam learning rate for the head

    Returns
    -------
    model : compiled tf.keras.Model ready for Phase 1 training
    """
    # Load pre-trained base without the ImageNet classification head
    base_model = MobileNetV2(
        input_shape  = input_shape,
        include_top  = False,
        weights      = "imagenet",
    )
    base_model.trainable = False    # freeze all base layers for Phase 1

    # Custom classification head
    x      = base_model.output
    x      = GlobalAveragePooling2D()(x)
    x      = BatchNormalization()(x)
    x      = Dense(128, activation="relu")(x)
    x      = Dropout(0.4)(x)
    output = Dense(1, activation="sigmoid")(x)   # binary: alert=0 / drowsy=1

    model = Model(inputs=base_model.input, outputs=output, name="MobileNetV2_Transfer")

    model.compile(
        optimizer = Adam(learning_rate=learning_rate),
        loss      = "binary_crossentropy",
        metrics   = ["accuracy"],
    )

    model.summary()
    return model


def unfreeze_for_fine_tuning(model, fine_tune_lr=1e-5):
    """
    Phase 2: unfreeze the top FINE_TUNE_AT layers of the MobileNetV2 base
    and recompile at a much lower learning rate.

    Call this after Phase 1 training is complete.

    Parameters
    ----------
    model        : the Phase-1 trained model returned by build_mobilenet()
    fine_tune_lr : float -- very small LR to avoid overwriting pre-trained weights

    Returns
    -------
    model : same model, partially unfrozen and recompiled for Phase 2
    """
    # ── How TF 2.15 builds functional models ─────────────────────────────────
    # When you call Model(inputs=base.input, outputs=output), TF 2.15 flattens
    # all of the base model's layers directly into model.layers — there is NO
    # nested sub-model object to find. All 154 MobileNetV2 layers appear in the
    # flat list alongside the 5 custom head layers.
    #
    # Strategy: locate where our custom head starts (GlobalAveragePooling2D),
    # treat everything before it as "the base", and unfreeze only the top
    # FINE_TUNE_AT of those base layers. Keep all BatchNorm frozen throughout.

    all_layers = model.layers

    # Find where the custom head starts — the first GlobalAveragePooling2D layer
    head_start = next(
        (i for i, l in enumerate(all_layers)
         if "global_average_pooling" in l.name.lower()),
        len(all_layers) - 5   # fallback: assume 5 head layers
    )

    base_layers  = all_layers[:head_start]   # MobileNetV2 layers (flat)
    total_base   = len(base_layers)
    freeze_up_to = max(0, total_base - FINE_TUNE_AT)

    print(f"\n[unfreeze] Total layers in model : {len(all_layers)}")
    print(f"[unfreeze] Base layers (flat)    : {total_base}")
    print(f"[unfreeze] Head starts at index  : {head_start}")
    print(f"[unfreeze] Freezing base[0:{freeze_up_to}], unfreezing base[{freeze_up_to}:{total_base}]")

    for i, layer in enumerate(all_layers):
        if i < freeze_up_to:
            layer.trainable = False          # keep early base layers frozen
        else:
            layer.trainable = True           # unfreeze top base + entire head
        # Always keep BatchNorm frozen to preserve running statistics
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer = Adam(learning_rate=fine_tune_lr),
        loss      = "binary_crossentropy",
        metrics   = ["accuracy"],
    )

    trainable = sum(1 for l in all_layers if l.trainable)
    print(f"[unfreeze] Trainable layers after unfreeze: {trainable} / {len(all_layers)}")
    return model
