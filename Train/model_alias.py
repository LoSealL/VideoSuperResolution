from VSR.Models import *

dncnn = DnCnn.DnCNN
espcn = Espcn.Espcn
srcnn = Srcnn.SRCNN
idn = Idn.InformationDistillationNetwork
rdn = Rdn.ResidualDenseNetwork
vespcn = Vespcn.VESPCN
dcscn = Dcscn.DCSCN


def get_model(name):
    return globals()[name]
