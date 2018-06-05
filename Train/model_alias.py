from VSR.Models import *


srcnn = Srcnn.SRCNN
espcn = Espcn.Espcn
vdsr = Vdsr.VDSR
drcn = Drcn.DRCN
dncnn = DnCnn.DnCNN
idn = Idn.InformationDistillationNetwork
rdn = Rdn.ResidualDenseNetwork
dcscn = Dcscn.DCSCN


def get_model(name):
    return globals()[name]
