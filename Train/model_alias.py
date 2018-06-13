from VSR.Models import *


srcnn = Srcnn.SRCNN
espcn = Espcn.ESPCN
vdsr = Vdsr.VDSR
drcn = Drcn.DRCN
dncnn = DnCnn.DnCNN
idn = Idn.InformationDistillationNetwork
rdn = Rdn.ResidualDenseNetwork
dcscn = Dcscn.DCSCN
lapsrn = LapSrn.LapSRN
drrn = Drrn.DRRN


def get_model(name):
    return globals()[name]
