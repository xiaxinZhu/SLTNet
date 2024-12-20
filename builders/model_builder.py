from model.SLTNet import SLTNet


def build_model(model_name, num_classes, ohem, early_loss):
    if model_name == 'SLTNet':
        return SLTNet(classes=num_classes, ohem=ohem, augment=early_loss)
