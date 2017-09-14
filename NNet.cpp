// NNet.cpp: implementation of the CNNet class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"

#include "FaxOMail.h"
#include "NNet.h"

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CNNLayer::CNNLayer( UINT nNodes, UINT nPrevLayerNodes ):m_Nodes(nNodes)
{
	m_Output = new double[nNodes+1] ;
	m_Output[nNodes] = 1.0 ;	// Bias part of Input to the Next layer.

	// If not input layer.. do the rest.
	if( nPrevLayerNodes != 0 ) {
		m_dError = new double[nNodes] ;

		m_Weights = new double*[nNodes] ;
		m_dWeights = new double*[nNodes] ;

		for( UINT i(0); i < nNodes; i++ ) {
			m_Weights[i] = new double[nPrevLayerNodes+1] ;
			m_dWeights[i] = new double[nPrevLayerNodes+1] ;
		}
	}
}

CNNLayer::~CNNLayer()
{
	delete[] m_dWeights ;
	delete[] m_Weights ;
	delete[] m_dError ;
	delete[] m_Output ;
}

CNNet::CNNet( UINT nLayers, UINT nNodes, ... ):m_nLayers(nLayers), 
	m_Eta(0.75), m_Eps(0.00005), m_Alpha(0.9), m_Bias(0.01)
{
	UINT Index(0), PrevArgument(0), Argument(nNodes) ; 

	m_Layers = (CNNLayer**)new int[nLayers] ;

	va_list marker;
	va_start( marker, nNodes );		// Initialize variable arguments.
	while( Index < nLayers )
	{
		m_Layers[Index++] = (CNNLayer*)new CNNLayer(Argument, PrevArgument) ;
		PrevArgument = Argument ;
		Argument = va_arg( marker, UINT );
	}
	va_end( marker );				// Reset variable arguments.

	// Allocate memory for m_DsOutput vector.
	m_DsOutput = new double[PrevArgument] ;
}

CNNet::~CNNet()
{
	delete[] m_Layers ;
	delete[] m_DsOutput ;
}

void CNNet::fRandomizeWeights()
{
	srand(1);
	double C(2.4) ;

	for( UINT nLayer(1); nLayer < m_nLayers; nLayer++ ) {
		double Hi( C/m_Layers[nLayer-1]->m_Nodes );
		double Lo( -C/m_Layers[nLayer-1]->m_Nodes );

		for( UINT i(0); i < m_Layers[nLayer]->m_Nodes; i++ ) {
			for( UINT j(0); j < m_Layers[nLayer-1]->m_Nodes; j++ ) {
				m_Layers[nLayer]->m_Weights[i][j] = ((double)rand()/RAND_MAX)*(Hi - Lo) + Lo;
				m_Layers[nLayer]->m_dWeights[i][j] = 0 ;
			}
			m_Layers[nLayer]->m_Weights[i][j] = m_Bias ;
			m_Layers[nLayer]->m_dWeights[i][j] = 0 ;
		}
	}
}

void CNNet::fSetIpVector( double* IpVec )
{
	CNNLayer* IpLayer = m_Layers[0] ;

	for( UINT i(0); i < IpLayer->m_Nodes; i++ )
		IpLayer->m_Output[i] = IpVec[i] ;
}

void CNNet::fSetDsOutput( double* OpVec )
{
	for( UINT i(0); i < m_Layers[m_nLayers-1]->m_Nodes; i++ )
		m_DsOutput[i] = OpVec[i] ;
}

double* CNNet::fGetOutput(UINT nLayer)
{
	ASSERT( nLayer > 0 && nLayer <= m_nLayers ) ;
	return m_Layers[nLayer-1]->m_Output ;
}

void CNNet::fPropagateNet()
{ 
	double Sum ;

	for( UINT nLayer(0); nLayer < m_nLayers-1; nLayer++ ) {
		CNNLayer* Lower = m_Layers[nLayer] ;
		CNNLayer* Upper = m_Layers[nLayer+1] ;
		
		for( UINT i(0); i < Upper->m_Nodes; i++ ) {
			Sum = 0;
			for( UINT j(0); j <= Lower->m_Nodes; j++ ) {
				Sum += Upper->m_Weights[i][j] * Lower->m_Output[j];
			}
			Upper->m_Output[i] = 1 / (1 + exp(-Sum));
		}
	}
}

double CNNet::fComputeOpError()
{
	double Out, Err;
	CNNLayer* OpLayer = m_Layers[m_nLayers-1] ;
   	m_Error = 0;
	
	for( UINT i(0); i < OpLayer->m_Nodes; i++ ) {
		Out = OpLayer->m_Output[i];
		Err = m_DsOutput[i] - Out;
		m_Error += 0.5*Err*Err;			// MSE

		OpLayer->m_dError[i] = Out * (1-Out) * Err;
	}
	return m_Error;
}

void CNNet::fBackPropagateError()
{
	double Out, Err;
	
	for( UINT nLayer(m_nLayers-1); nLayer > 1; nLayer-- ) {
		CNNLayer* Lower = m_Layers[nLayer-1] ;
		CNNLayer* Upper = m_Layers[nLayer] ;
	
		for( UINT i(0); i < Lower->m_Nodes; i++ ) {
			Out = Lower->m_Output[i];
			Err = 0;

			for( UINT j(0); j < Upper->m_Nodes; j++ ) {
				Err += Upper->m_Weights[j][i] * Upper->m_dError[j];
			}
			Lower->m_dError[i] =  Out * (1-Out) * Err;
		}
	}

	// Adjust the weights of the layers based on the value of m_dError[] obtained.
	fAdjustWeights() ;
}

void CNNet::fAdjustWeights()
{
	double Out, Err, dWeight;
   
	for( UINT nLayer(1); nLayer < m_nLayers; nLayer++ ) {
		for( UINT i(0); i < m_Layers[nLayer]->m_Nodes; i++ ) {
			for( UINT j(0); j <= m_Layers[nLayer-1]->m_Nodes; j++ ) {
				Out = m_Layers[nLayer-1]->m_Output[j];
				Err = m_Layers[nLayer]->m_dError[i];
				dWeight = m_Layers[nLayer]->m_dWeights[i][j];
				m_Layers[nLayer]->m_Weights[i][j] += m_Eta * Err * Out + m_Alpha * dWeight;
				m_Layers[nLayer]->m_dWeights[i][j] = m_Eta * Err * Out;
			}
		}
	}
}

// Member functions for loading wts. from and storing wts. to Archive ( File ).
void CNNet::fStoreWeights(CArchive & ar)
{
	for( UINT nLayer(1); nLayer < m_nLayers; nLayer++ )
		for( UINT i(0); i < m_Layers[nLayer]->m_Nodes; i++ )
			for( UINT j(0); j <= m_Layers[nLayer-1]->m_Nodes; j++ )
				ar << m_Layers[nLayer]->m_Weights[i][j] ;
}

void CNNet::fLoadWeights(CArchive & ar)
{
	for( UINT nLayer(1); nLayer < m_nLayers; nLayer++ )
		for( UINT i(0); i < m_Layers[nLayer]->m_Nodes; i++ )
			for( UINT j(0); j <= m_Layers[nLayer-1]->m_Nodes; j++ ) {
				ar >> m_Layers[nLayer]->m_Weights[i][j] ;
				m_Layers[nLayer]->m_dWeights[i][j] = 0 ; 
			}
}

void CNNet::fTrainNet()
{
	fRandomizeWeights();
	fPropagateNet() ;
	
	while( fComputeOpError() > m_Eps ) {
		fBackPropagateError() ;
		fPropagateNet() ;
	}
}
