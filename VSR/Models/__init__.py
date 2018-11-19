from . import (Srcnn, Espcn, Dcscn, DnCnn, Vdsr, Drcn, Drrn, LapSrn, MemNet,
               Edsr, Idn, Rdn, Dbpn, Carn, Rcan, SrGan, Vespcn)

__all__ = ['get_model', 'list_supported_models']

alias = {
    'srcnn': Srcnn.SRCNN,
    'espcn': Espcn.ESPCN,
    'vdsr': Vdsr.VDSR,
    'drcn': Drcn.DRCN,
    'dncnn': DnCnn.DnCNN,
    'idn': Idn.InformationDistillationNetwork,
    'rdn': Rdn.ResidualDenseNetwork,
    'dcscn': Dcscn.DCSCN,
    'lapsrn': LapSrn.LapSRN,
    'drrn': Drrn.DRRN,
    'memnet': MemNet.MEMNET,
    'dbpn': Dbpn.DBPN,
    'edsr': Edsr.EDSR,
    'srgan': SrGan.SRGAN,
    'carn': Carn.CARN,
    'rcan': Rcan.RCAN,
    'vespcn': Vespcn.VESPCN,
}


def get_model(name):
    return alias[name]


def list_supported_models():
    return alias.keys()
