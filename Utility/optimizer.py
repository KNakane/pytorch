import torch
import torch.optim as optim

def get_optimizer(name=None,
                  model=None,
                  lr=0.001):
    assert name is not None
    if name == "SGD":
        return optim.SGD(params=model.parameters(),
                         lr=lr, momentum=0)
    elif name == "Momentum":
        return optim.SGD(params=model.parameters(),
                         lr=lr, momentum=0.9)
    elif name == "Adadelta":
        return optim.Adadelta(params=model.parameters(),
                              lr=lr, rho=0.9, eps=1e-06, weight_decay=0)
    elif name == "Adagrad":
        return optim.Adagrad(params=model.parameters(),
                             lr=lr, lr_decay=0, weight_decay=0,
                             initial_accumulator_value=0, eps=1e-10)
    elif name == "Adam":
        return optim.Adam(params=model.parameters(),
                          lr=lr, betas=(0.9, 0.999), eps=1e-08,
                          weight_decay=0, amsgrad=False)
    elif name == "RMSprop":
        return optim.RMSprop(params=model.parameters(),
                             lr=lr, alpha=0.99, eps=1e-08,
                             weight_decay=0, momentum=0, centered=False)
    elif name == "Adamax":
        return optim.Adamax(params=model.parameters(),
                            lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif name == "RAdam":
        return optim.RAdam(params=model.parameters(),
                           lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        raise NotImplementedError()