"""Africa-first payment and banking connectors for APG.

Available connectors:
  MPESAConnector       — Safaricom MPESA Daraja 2.0 (KE, TZ, UG, GH)
  EquityBankConnector  — Equity Bank Group API (KE, UG, TZ, RW, DRC, SS)
  KCBConnector         — KCB Bank (KE, UG, TZ, RW, ET, SS, BI)
  MTNConnector         — MTN Mobile Money / MoMo (NG, GH, UG, CM, CI, ZM, ...)
  AirtelConnector      — Airtel Money (KE, UG, TZ, RW, ZM, MG, CD, ...)
  OrangeConnector      — Orange Money (CI, SN, CM, ML, BF, MG, NE, ...)
  WaveConnector        — Wave Mobile Money (SN, CI, ML, BF, GN)
  MShwariConnector     — M-Shwari savings & loans via Daraja (KE only)
"""
from .mpesa_connector import MPESAConnector, MPESAConfiguration, mpesa_connector_from_env
from .equity_connector import EquityBankConnector, EquityBankConfiguration, equity_connector_from_env
from .kcb_connector import KCBConnector, KCBConfiguration, kcb_connector_from_env
from .mtn_connector import MTNConnector, MTNConfiguration, mtn_connector_from_env
from .airtel_connector import AirtelConnector, AirtelConfiguration, airtel_connector_from_env
from .orange_connector import OrangeConnector, OrangeConfiguration, orange_connector_from_env
from .wave_connector import WaveConnector, WaveConfiguration, wave_connector_from_env
from .mshwari_connector import MShwariConnector, MShwariConfiguration, mshwari_connector_from_env

__all__ = [
	"MPESAConnector", "MPESAConfiguration", "mpesa_connector_from_env",
	"EquityBankConnector", "EquityBankConfiguration", "equity_connector_from_env",
	"KCBConnector", "KCBConfiguration", "kcb_connector_from_env",
	"MTNConnector", "MTNConfiguration", "mtn_connector_from_env",
	"AirtelConnector", "AirtelConfiguration", "airtel_connector_from_env",
	"OrangeConnector", "OrangeConfiguration", "orange_connector_from_env",
	"WaveConnector", "WaveConfiguration", "wave_connector_from_env",
	"MShwariConnector", "MShwariConfiguration", "mshwari_connector_from_env",
]
