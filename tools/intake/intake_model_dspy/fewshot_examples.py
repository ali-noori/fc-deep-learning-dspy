"""Few-shot pairs: one model request -> four copied fields."""

from __future__ import annotations

# (user_message, architecture_description, model_selection, in_feature, n_classes)
# Each output is a copy of words the user wrote. Empty means they did not write that fact.
# Total few-shots: 29
FEWSHOT_PAIRS: list[tuple[str, str, str, str, str]] = [
    # 1. Layer wording copied as written. The two counts are the values in the last sentence.
    (
        "Create a simple CNN like plugins/models/cnn.py for 28x28 images, "
        "in_features channels, n_classes classes. Conv2d in_features->10 kernel=5 no padding. "
        "MaxPool2d(2) ReLU. Conv2d 10->20 kernel=5 no padding. Dropout2d. MaxPool2d(2) ReLU. "
        "Flatten to 320. Linear 320->50 ReLU. Dropout. Linear 50->n_classes. "
        "Return the Linear scores only. Do not use softmax or log_softmax. "
        "Also consider that we have 1 input feature and the number of classes is 10.",
        "Create a simple CNN like plugins/models/cnn.py for 28x28 images, "
        "in_features channels, n_classes classes. Conv2d in_features->10 kernel=5 no padding. "
        "MaxPool2d(2) ReLU. Conv2d 10->20 kernel=5 no padding. Dropout2d. MaxPool2d(2) ReLU. "
        "Flatten to 320. Linear 320->50 ReLU. Dropout. Linear 50->n_classes. "
        "Return the Linear scores only. Do not use softmax or log_softmax.",
        "",
        "1",
        "10",
    ),
    # 2. Bare model id. The class value is the word the user wrote.
    (
        "efficientnet_b0 and number of input is 1 and number of class is ten.",
        "",
        "efficientnet_b0",
        "1",
        "ten",
    ),
    # 3. key: value lines. Field names in_features and n_classes.
    (
        "name: cnn.py\nin_features: 1\nn_classes: 10",
        "",
        "cnn.py",
        "1",
        "10",
    ),
    # 4. Plain sentences: channels and classes, no field names.
    (
        "resnet18.py. 3 input channels. 10 classes.",
        "",
        "resnet18.py",
        "3",
        "10",
    ),
    # 5. "Input features" and "Output classes".
    (
        "mlp.py. Input features 784. Output classes 2.",
        "",
        "mlp.py",
        "784",
        "2",
    ),
    # 6. Short keys in_feature and n_classes, name without .py.
    (
        "mobilenetv3_small. in_feature 1. n_classes 10.",
        "",
        "mobilenetv3_small",
        "1",
        "10",
    ),
    # 7. Short layer sentence, counts after it.
    (
        "Input size 784, then Linear 784->128 ReLU, Dropout, Linear 128->n_classes. "
        "in_features 784. n_classes 5.",
        "Input size 784, then Linear 784->128 ReLU, Dropout, Linear 128->n_classes.",
        "",
        "784",
        "5",
    ),
    # 8. Tabular plugin, "input features" and "classes".
    (
        "tabnet.py. 32 input features. 4 classes.",
        "",
        "tabnet.py",
        "32",
        "4",
    ),
    # 9. Saved custom file name, copied as written.
    (
        "customized_model_v2.py. 1 input feature. 10 classes.",
        "",
        "customized_model_v2.py",
        "1",
        "10",
    ),
    # 10. Counts written in a sentence, then the layer wording.
    (
        "number of input is one and number of class is 10. "
        "Conv2d in_features->8 kernel=3, ReLU, Flatten, Linear to n_classes.",
        "Conv2d in_features->8 kernel=3, ReLU, Flatten, Linear to n_classes.",
        "",
        "one",
        "10",
    ),
    # 11. One layer per line. Count lines are not part of the description.
    (
        "in_features 1\n"
        "n_classes 10\n"
        "Conv2d: in_features->10, kernel=5, no padding\n"
        "MaxPool2d(2) -> ReLU\n"
        "Flatten to 320\n"
        "Linear: 320->n_classes",
        "Conv2d: in_features->10, kernel=5, no padding\n"
        "MaxPool2d(2) -> ReLU\n"
        "Flatten to 320\n"
        "Linear: 320->n_classes",
        "",
        "1",
        "10",
    ),
    # 12. Layer wording, then the two labeled values.
    (
        "First a Linear from in_features to 64 with ReLU, then Dropout, "
        "then Linear from 64 to n_classes. in_features 20. n_classes 3.",
        "First a Linear from in_features to 64 with ReLU, then Dropout, "
        "then Linear from 64 to n_classes.",
        "",
        "20",
        "3",
    ),
    # 13. Layer lines copied together. Count lines are the two values.
    (
        "- Conv2d in_features->8 kernel=3\n"
        "- ReLU\n"
        "- Linear to n_classes\n"
        "in_feature 1\n"
        "n_classes 10",
        "- Conv2d in_features->8 kernel=3\n"
        "- ReLU\n"
        "- Linear to n_classes",
        "",
        "1",
        "10",
    ),
    # 14. Semicolon-separated values and layer wording.
    (
        "features=16; classes=4; Linear 16->32 ReLU; Dropout; Linear 32->n_classes",
        "Linear 16->32 ReLU; Dropout; Linear 32->n_classes",
        "",
        "16",
        "4",
    ),
    # 15. Full path and the n_class value, copied as written.
    (
        "name: plugins/models/pre-trained_models/cnn_architectures/efficientnet_b0.py\n"
        "n_class: 10\n"
        "in_features: 1",
        "",
        "plugins/models/pre-trained_models/cnn_architectures/efficientnet_b0.py",
        "1",
        "10",
    ),
    # 16. The name CNN is copied as written.
    (
        "name: CNN\n"
        "n_class: 10\n"
        "in_features: 1",
        "",
        "CNN",
        "1",
        "10",
    ),
    # 17. The name and the two values, written in a different order.
    (
        "7 classes. ft_transformer.py. 16 input features.",
        "",
        "ft_transformer.py",
        "16",
        "7",
    ),
    # 18. Bullet list.
    (
        "- name: tabtransformer.py\n"
        "- n_classes: 2\n"
        "- in_features: 32",
        "",
        "tabtransformer.py",
        "32",
        "2",
    ),
    # 19. JSON values copied into the matching fields.
    (
        '{"name": "resnet18.py", "in_features": 3, "n_classes": 100}',
        "",
        "resnet18.py",
        "3",
        "100",
    ),
    # 20. Counts first, then the path the user wrote.
    (
        "n_classes=10; in_features=1; model plugins\\models\\c_f_test2.py",
        "",
        "plugins\\models\\c_f_test2.py",
        "1",
        "10",
    ),
    # 21. Plain sentences. The counts are the words three and ten.
    (
        "resnet18.py. three input channels. ten classes.",
        "",
        "resnet18.py",
        "three",
        "ten",
    ),
    # 22. A multi-word number is copied as the whole phrase.
    (
        "mlp.py. Input features seven hundred eighty-four. Output classes two.",
        "",
        "mlp.py",
        "seven hundred eighty-four",
        "two",
    ),
    # 23. Hyphenated word for the input count.
    (
        "tabnet.py. thirty-two input features. four classes.",
        "",
        "tabnet.py",
        "thirty-two",
        "four",
    ),
    # 24. Both counts are words.
    (
        "customized_model_v2.py. one input feature. ten classes.",
        "",
        "customized_model_v2.py",
        "one",
        "ten",
    ),
    # 25. Layer wording, then word values.
    (
        "First a Linear from in_features to 64 with ReLU, then Dropout, "
        "then Linear from 64 to n_classes. in_features twenty. n_classes three.",
        "First a Linear from in_features to 64 with ReLU, then Dropout, "
        "then Linear from 64 to n_classes.",
        "",
        "twenty",
        "three",
    ),
    # 26. Key values written as words.
    (
        "name: CNN\n"
        "n_class: ten\n"
        "in_features: one",
        "",
        "CNN",
        "one",
        "ten",
    ),
    # 27. Word values written in a different order.
    (
        "seven classes. ft_transformer.py. sixteen input features.",
        "",
        "ft_transformer.py",
        "sixteen",
        "seven",
    ),
    # 28. Bullet list with word values.
    (
        "- name: tabtransformer.py\n"
        "- n_classes: two\n"
        "- in_features: thirty-two",
        "",
        "tabtransformer.py",
        "thirty-two",
        "two",
    ),
    # 29. Word values, then the path the user wrote.
    (
        "n_classes=ten; in_features=one; model plugins\\models\\c_f_test2.py",
        "",
        "plugins\\models\\c_f_test2.py",
        "one",
        "ten",
    ),
]
