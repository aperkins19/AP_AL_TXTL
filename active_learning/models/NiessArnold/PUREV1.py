# Python model ODEs from antimony file

import numpy as np

def model(y, t, params):

	Cgtp = y[0]
	Cgdp = y[1]
	Cgmp = y[2]
	Catp = y[3]
	Camp = y[4]
	Cribo70s = y[5]
	Cmrna = y[6]
	Cp = y[7]
	Cefg = y[8]
	Cefggdp = y[9]
	Cefggtp = y[10]
	Ceftu = y[11]
	Ceftugdp = y[12]
	Ceftugtp = y[13]
	Ct3 = y[14]
	Cif = y[15]
	Cifgtp = y[16]
	Cribo30s = y[17]
	Cribo50s = y[18]
	Cr30SIF = y[19]
	Cr70SIC = y[20]
	CfmetT = y[21]
	Caa = y[22]
	Caat = y[23]
	Ct = y[24]
	Cpi = y[25]
	Cppi = y[26]
	Cutp = y[27]
	Cctp = y[28]
	Cump = y[29]
	Ccmp = y[30]
	C1 = y[31]
	C2 = y[32]
	C3 = y[33]
	C4 = y[34]
	C5 = y[35]
	C6 = y[36]
	C7 = y[37]
	C8 = y[38]
	C9 = y[39]
	C10 = y[40]
	C11 = y[41]
	C12 = y[42]
	C13 = y[43]
	C14 = y[44]
	C15 = y[45]
	C16 = y[46]
	C17 = y[47]
	C18 = y[48]
	C19 = y[49]
	C20 = y[50]
	C21 = y[51]
	C22 = y[52]
	C23 = y[53]
	C24 = y[54]
	C25 = y[55]
	C26 = y[56]
	C27 = y[57]
	C28 = y[58]
	C29 = y[59]
	C30 = y[60]
	C31 = y[61]
	C32 = y[62]

	kfefg1 = params[0]
	krefg1 = params[1]
	kfefg2 = params[2]
	krefg2 = params[3]
	kfeftu1 = params[4]
	kreftu1 = params[5]
	kfeftu2 = params[6]
	kreftu2 = params[7]
	keftsf = params[8]
	Ceftst = params[9]
	Keqeftu = params[10]
	Kmb = params[11]
	Kma = params[12]
	keftsr = params[13]
	Kmp = params[14]
	Kmq = params[15]
	Kiq = params[16]
	Kia = params[17]
	Vmaxars = params[18]
	Kmarsaa = params[19]
	Kmarsatp = params[20]
	Kmarst = params[21]
	kft3form = params[22]
	krt3form = params[23]
	ktle = params[24]
	Kmt3 = params[25]
	Kmefggtp = params[26]
	ktlt = params[27]
	Crf = params[28]
	Kmrk = params[29]
	Kmgtp = params[30]
	konribo = params[31]
	koffribo = params[32]
	kon1 = params[33]
	koff1 = params[34]
	kon2 = params[35]
	koff2 = params[36]
	kTLI70SIC = params[37]
	KmfmetT = params[38]
	Kmmrna = params[39]
	Km50s = params[40]
	kTLIIF2D = params[41]
	Vmaxtx = params[42]
	Kmd = params[43]
	dt = params[44]
	Kda = params[45]
	Kdg = params[46]
	Kdu = params[47]
	Kdc = params[48]
	kmrnadeg = params[49]
	Cribo30S = params[50]
	Cribo50S = params[51]

	derivs = [
	- 1.0 * (kfefg2*Cefg*Cgtp-krefg2*Cefggtp) - 1.0 * (kfeftu1*Ceftu*Cgtp-kreftu1*Ceftugtp) - 1.0 * (keftsf*Ceftst*(Ceftugdp*Cgtp-Cgdp*Ceftugtp/Keqeftu)/(Kmb*Ceftugdp+Kma*Cgtp+keftsf*Ceftst*Cgdp*Ceftugtp/(keftsr*Ceftst*Keqeftu)+keftsf*Ceftst*Kmp*Ceftugtp/(keftsr*Ceftst*Keqeftu)+keftsf*Ceftst*Kmq*Cgdp/(keftsr*Ceftst*Keqeftu)+Ceftugdp*Cgtp+Kma*Cgtp*Ceftugtp/Kiq+keftsf*Ceftst*Kmq*Ceftugdp*Cgdp/(keftsr*Ceftst*Keqeftu*Kia))) - 1.0 * (ktlt*Crf/(1+Kmrk/C32+Kmgtp/Cgtp+Kmrk*Kmgtp/(C32*Cgtp))) - 1.0 * (kon1*Cgtp*Cif) + 1.0 * (koff1*Cifgtp) - 24.0 * (Vmaxtx/(1+Kmd/dt+24*(Kda/Catp)+24*(Kdg/Cgtp)+24*(Kdu/Cutp)+24*(Kdc/Cctp))),
	+ 1.0 * (kfefg1*Cefggdp-krefg1*Cefg*Cgdp) - 1.0 * (kfeftu2*Ceftu*Cgdp-kreftu2*Ceftugdp) + 1.0 * (keftsf*Ceftst*(Ceftugdp*Cgtp-Cgdp*Ceftugtp/Keqeftu)/(Kmb*Ceftugdp+Kma*Cgtp+keftsf*Ceftst*Cgdp*Ceftugtp/(keftsr*Ceftst*Keqeftu)+keftsf*Ceftst*Kmp*Ceftugtp/(keftsr*Ceftst*Keqeftu)+keftsf*Ceftst*Kmq*Cgdp/(keftsr*Ceftst*Keqeftu)+Ceftugdp*Cgtp+Kma*Cgtp*Ceftugtp/Kiq+keftsf*Ceftst*Kmq*Ceftugdp*Cgdp/(keftsr*Ceftst*Keqeftu*Kia))) + 1.0 * (ktlt*Crf/(1+Kmrk/C32+Kmgtp/Cgtp+Kmrk*Kmgtp/(C32*Cgtp))) + 1.0 * (kTLIIF2D*Cr70SIC),
	+ 24.0 * (kmrnadeg*Cmrna),
	- 1.0 * (Vmaxars*1/(1+Kmarsaa/Caa+Kmarsatp/Catp+Kmarst/Ct)) - 24.0 * (Vmaxtx/(1+Kmd/dt+24*(Kda/Catp)+24*(Kdg/Cgtp)+24*(Kdu/Cutp)+24*(Kdc/Cctp))),
	+ 1.0 * (Vmaxars*1/(1+Kmarsaa/Caa+Kmarsatp/Catp+Kmarst/Ct)) + 24.0 * (kmrnadeg*Cmrna),
	+ 1.0 * (ktlt*Crf/(1+Kmrk/C32+Kmgtp/Cgtp+Kmrk*Kmgtp/(C32*Cgtp))) + 1.0 * (konribo*Cribo30s*Cribo50s) - 1.0 * (koffribo*Cribo70s),
	+ 1.0 * (ktlt*Crf/(1+Kmrk/C32+Kmgtp/Cgtp+Kmrk*Kmgtp/(C32*Cgtp))) - 1.0 * (kTLI70SIC*Cr30SIF/(1+KmfmetT/CfmetT+Kmmrna/Cmrna+Km50s/Cribo50s+KmfmetT*Kmmrna/(CfmetT*Cmrna))) + 1.0 * (Vmaxtx/(1+Kmd/dt+24*(Kda/Catp)+24*(Kdg/Cgtp)+24*(Kdu/Cutp)+24*(Kdc/Cctp))) - 1.0 * (kmrnadeg*Cmrna),
	+ 1.0 * (ktlt*Crf/(1+Kmrk/C32+Kmgtp/Cgtp+Kmrk*Kmgtp/(C32*Cgtp))),
	+ 1.0 * (kfefg1*Cefggdp-krefg1*Cefg*Cgdp) - 1.0 * (kfefg2*Cefg*Cgtp-krefg2*Cefggtp),
	- 1.0 * (kfefg1*Cefggdp-krefg1*Cefg*Cgdp) + 1.0 * (ktle*C1/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C2/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C3/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C4/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C5/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C6/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C7/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C8/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C9/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C10/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C11/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C12/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C13/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C14/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C15/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C16/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C17/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C18/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C19/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C20/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C21/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C22/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C23/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C24/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C25/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C26/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C27/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C28/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C29/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C30/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C31/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (kfefg2*Cefg*Cgtp-krefg2*Cefggtp) - 1.0 * (ktle*C1/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C2/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C3/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C4/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C5/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C6/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C7/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C8/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C9/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C10/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C11/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C12/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C13/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C14/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C15/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C16/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C17/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C18/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C19/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C20/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C21/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C22/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C23/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C24/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C25/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C26/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C27/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C28/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C29/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C30/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C31/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	- 1.0 * (kfeftu1*Ceftu*Cgtp-kreftu1*Ceftugtp) - 1.0 * (kfeftu2*Ceftu*Cgdp-kreftu2*Ceftugdp),
	+ 1.0 * (kfeftu2*Ceftu*Cgdp-kreftu2*Ceftugdp) - 1.0 * (keftsf*Ceftst*(Ceftugdp*Cgtp-Cgdp*Ceftugtp/Keqeftu)/(Kmb*Ceftugdp+Kma*Cgtp+keftsf*Ceftst*Cgdp*Ceftugtp/(keftsr*Ceftst*Keqeftu)+keftsf*Ceftst*Kmp*Ceftugtp/(keftsr*Ceftst*Keqeftu)+keftsf*Ceftst*Kmq*Cgdp/(keftsr*Ceftst*Keqeftu)+Ceftugdp*Cgtp+Kma*Cgtp*Ceftugtp/Kiq+keftsf*Ceftst*Kmq*Ceftugdp*Cgdp/(keftsr*Ceftst*Keqeftu*Kia))) + 1.0 * (ktle*C1/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C2/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C3/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C4/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C5/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C6/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C7/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C8/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C9/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C10/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C11/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C12/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C13/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C14/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C15/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C16/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C17/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C18/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C19/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C20/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C21/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C22/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C23/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C24/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C25/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C26/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C27/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C28/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C29/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C30/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C31/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (kfeftu1*Ceftu*Cgtp-kreftu1*Ceftugtp) + 1.0 * (keftsf*Ceftst*(Ceftugdp*Cgtp-Cgdp*Ceftugtp/Keqeftu)/(Kmb*Ceftugdp+Kma*Cgtp+keftsf*Ceftst*Cgdp*Ceftugtp/(keftsr*Ceftst*Keqeftu)+keftsf*Ceftst*Kmp*Ceftugtp/(keftsr*Ceftst*Keqeftu)+keftsf*Ceftst*Kmq*Cgdp/(keftsr*Ceftst*Keqeftu)+Ceftugdp*Cgtp+Kma*Cgtp*Ceftugtp/Kiq+keftsf*Ceftst*Kmq*Ceftugdp*Cgdp/(keftsr*Ceftst*Keqeftu*Kia))) - 1.0 * (kft3form*Ceftugtp*Caat-krt3form*Ct3),
	+ 1.0 * (kft3form*Ceftugtp*Caat-krt3form*Ct3) - 1.0 * (ktle*C1/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C2/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C3/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C4/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C5/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C6/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C7/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C8/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C9/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C10/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C11/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C12/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C13/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C14/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C15/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C16/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C17/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C18/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C19/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C20/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C21/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C22/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C23/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C24/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C25/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C26/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C27/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C28/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C29/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C30/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C31/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	- 1.0 * (kon1*Cgtp*Cif) + 1.0 * (koff1*Cifgtp) + 1.0 * (kTLIIF2D*Cr70SIC),
	+ 1.0 * (kon1*Cgtp*Cif) - 1.0 * (koff1*Cifgtp) - 1.0 * (kon2*Cribo30s*Cifgtp) + 1.0 * (koff2*Cr30SIF),
	- 1.0 * (konribo*Cribo30s*Cribo50s) + 1.0 * (koffribo*Cribo70s) - 1.0 * (kon2*Cribo30s*Cifgtp) + 1.0 * (koff2*Cr30SIF),
	- 1.0 * (konribo*Cribo30s*Cribo50s) + 1.0 * (koffribo*Cribo70s) - 1.0 * (kTLI70SIC*Cr30SIF/(1+KmfmetT/CfmetT+Kmmrna/Cmrna+Km50s/Cribo50s+KmfmetT*Kmmrna/(CfmetT*Cmrna))),
	+ 1.0 * (kon2*Cribo30s*Cifgtp) - 1.0 * (koff2*Cr30SIF) - 1.0 * (kTLI70SIC*Cr30SIF/(1+KmfmetT/CfmetT+Kmmrna/Cmrna+Km50s/Cribo50s+KmfmetT*Kmmrna/(CfmetT*Cmrna))),
	+ 1.0 * (kTLI70SIC*Cr30SIF/(1+KmfmetT/CfmetT+Kmmrna/Cmrna+Km50s/Cribo50s+KmfmetT*Kmmrna/(CfmetT*Cmrna))) - 1.0 * (kTLIIF2D*Cr70SIC),
	- 1.0 * (kTLI70SIC*Cr30SIF/(1+KmfmetT/CfmetT+Kmmrna/Cmrna+Km50s/Cribo50s+KmfmetT*Kmmrna/(CfmetT*Cmrna))),
	- 1.0 * (Vmaxars*1/(1+Kmarsaa/Caa+Kmarsatp/Catp+Kmarst/Ct)),
	+ 1.0 * (Vmaxars*1/(1+Kmarsaa/Caa+Kmarsatp/Catp+Kmarst/Ct)) - 1.0 * (kft3form*Ceftugtp*Caat-krt3form*Ct3),
	- 1.0 * (Vmaxars*1/(1+Kmarsaa/Caa+Kmarsatp/Catp+Kmarst/Ct)) + 1.0 * (ktle*C1/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C2/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C3/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C4/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C5/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C6/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C7/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C8/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C9/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C10/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C11/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C12/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C13/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C14/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C15/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C16/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C17/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C18/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C19/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C20/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C21/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C22/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C23/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C24/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C25/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C26/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C27/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C28/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C29/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C30/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktle*C31/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 2.0 * (ktle*C1/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C2/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C3/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C4/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C5/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C6/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C7/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C8/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C9/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C10/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C11/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C12/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C13/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C14/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C15/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C16/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C17/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C18/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C19/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C20/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C21/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C22/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C23/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C24/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C25/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C26/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C27/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C28/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C29/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C30/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 2.0 * (ktle*C31/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (ktlt*Crf/(1+Kmrk/C32+Kmgtp/Cgtp+Kmrk*Kmgtp/(C32*Cgtp))) + 1.0 * (kTLIIF2D*Cr70SIC),
	+ 1.0 * (Vmaxars*1/(1+Kmarsaa/Caa+Kmarsatp/Catp+Kmarst/Ct)) + 96.0 * (Vmaxtx/(1+Kmd/dt+24*(Kda/Catp)+24*(Kdg/Cgtp)+24*(Kdu/Cutp)+24*(Kdc/Cctp))),
	- 24.0 * (Vmaxtx/(1+Kmd/dt+24*(Kda/Catp)+24*(Kdg/Cgtp)+24*(Kdu/Cutp)+24*(Kdc/Cctp))),
	- 24.0 * (Vmaxtx/(1+Kmd/dt+24*(Kda/Catp)+24*(Kdg/Cgtp)+24*(Kdu/Cutp)+24*(Kdc/Cctp))),
	+ 24.0 * (kmrnadeg*Cmrna),
	+ 24.0 * (kmrnadeg*Cmrna),
	- 1.0 * (ktle*C1/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) + 1.0 * (kTLIIF2D*Cr70SIC),
	+ 1.0 * (ktle*C1/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C2/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C2/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C3/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C3/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C4/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C4/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C5/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C5/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C6/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C6/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C7/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C7/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C8/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C8/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C9/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C9/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C10/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C10/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C11/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C11/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C12/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C12/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C13/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C13/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C14/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C14/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C15/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C15/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C16/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C16/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C17/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C17/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C18/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C18/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C19/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C19/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C20/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C20/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C21/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C21/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C22/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C22/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C23/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C23/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C24/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C24/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C25/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C25/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C26/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C26/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C27/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C27/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C28/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C28/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C29/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C29/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C30/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C30/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktle*C31/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)),
	+ 1.0 * (ktle*C31/(1+Kmt3/Ct3+Kmefggtp/Cefggtp)) - 1.0 * (ktlt*Crf/(1+Kmrk/C32+Kmgtp/Cgtp+Kmrk*Kmgtp/(C32*Cgtp)))]
	return derivs

keysVar = ['Cgtp','Cgdp','Cgmp','Catp','Camp','Cribo70s','Cmrna','Cp','Cefg','Cefggdp','Cefggtp','Ceftu','Ceftugdp','Ceftugtp','Ct3','Cif','Cifgtp','Cribo30s','Cribo50s','Cr30SIF','Cr70SIC','CfmetT','Caa','Caat','Ct','Cpi','Cppi','Cutp','Cctp','Cump','Ccmp','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31','C32']
valuesVar = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,31.9,0.0,1.0,236.69,0.0,0.0,1.0,10.0,0.0,0.0,0.0,0.0,0.0,20.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
dictVar = dict(zip(keysVar,valuesVar))

keysPar = ['kfefg1','krefg1','kfefg2','krefg2','kfeftu1','kreftu1','kfeftu2','kreftu2','keftsf','Ceftst','Keqeftu','Kmb','Kma','keftsr','Kmp','Kmq','Kiq','Kia','Vmaxars','Kmarsaa','Kmarsatp','Kmarst','kft3form','krt3form','ktle','Kmt3','Kmefggtp','ktlt','Crf','Kmrk','Kmgtp','konribo','koffribo','kon1','koff1','kon2','koff2','kTLI70SIC','KmfmetT','Kmmrna','Km50s','kTLIIF2D','Vmaxtx','Kmd','dt','Kda','Kdg','Kdu','Kdc','kmrnadeg','Cribo30S','Cribo50S']
valuesPar = [100.0,27.0,10.0,100.0,0.2,0.025,0.9,0.0017,30.0,5.0,0.19,50.0,2.5,10.0,3.0,1.0,1.0,5.6,5.95,20.0,100.0,0.5,50.0,1.0,24.0,0.4,0.22,0.5,37.6,0.0083,20.0,0.2385,0.0045,1000.0,1.0,80.0,1.0,0.1,0.05,0.009,0.012,1.5,0.000865,0.0063,0.001,76.0,76.0,33.0,34.0,0.000138,39.82,39.82]
dictPar = dict(zip(keysPar,valuesPar))