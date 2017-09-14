// NNet.h: interface for the CNNet class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_NNET_H__B6277702_B36A_11D1_8BDF_444553540000__INCLUDED_)
#define AFX_NNET_H__B6277702_B36A_11D1_8BDF_444553540000__INCLUDED_

#if _MSC_VER >= 1000
#pragma once
#endif // _MSC_VER >= 1000

#include <math.h>
#include <stdarg.h>

class CNNLayer
{
public:
	double* m_Output ;
	double* m_dError ;
	double** m_Weights ;
	double** m_dWeights ;
	UINT m_Nodes ;
	CNNLayer(UINT nNodes, UINT nPrevLayerNodes) ;
	virtual ~CNNLayer() ;
};

class CNNet  
{
public:
	void fTrainNet();
	void fLoadWeights(CArchive& ar);
	void fStoreWeights(CArchive& ar);
	double m_Error;

	void fBackPropagateError();
	void fComputeOpError();
	void fPropagateNet();
	double* fGetOutput( UINT nLayer );
	void fSetDsOutput(double* OpVec);
	void fSetIpVector(double* IpVec);
	void fRandomizeWeights();
	CNNet(UINT nLayers, UINT nNodes, ...);
	virtual ~CNNet();

private:
	void fAdjustWeights();

	double* m_DsOutput;

	double m_Bias;
	double m_Eps;
	double m_Eta;
	double m_Alpha;

	UINT m_nLayers;
	CNNLayer** m_Layers;
};

#endif // !defined(AFX_NNET_H__B6277702_B36A_11D1_8BDF_444553540000__INCLUDED_)
