FeatureCloud model plugin contract (always apply):

- Define exactly one class named Model that subclasses torch.nn.Module.
- Constructor must be __init__(self, n_classes, in_features).
- Implement forward(self, x).
- Include torch / nn imports. The app loads this file by filename and looks up class Model.
- Copy this interface from retrieved examples. Do not copy an example's layers if the user asked for a different architecture.
