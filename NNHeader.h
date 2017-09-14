//	NNHeader.h : The header file containing the declarations for NN..
//

#if !defined(NNHEADER_H)
#define NNHEADER_H

#if _MSC_VER >= 1000
#pragma once
#endif // _MSC_VER >= 1000

#define XOffset			50
#define YOffset			50
#define START_VALUE     200		// Starting of ID value in the string table
#define END_VALUE       292		// Ending of ID value in the string table

#define nR				15		// Rows in the Input pattern.
#define nS				15		// Cols in the Input pattern.
#define MAX_INPUT		nR*nS	// An RxS matrix
#define MAX_INPUT1		6*nR-2	// An RxS matrix
#define MAX_HID_NODES	20//45
#define MAX_HID_NODES1	40
#define MAX_OUTPUT      8//nR*nS//END_VALUE-START_VALUE+1
#define OP_LAYER		3
#define HI				0.9
#define LO				0.1
#define a				1.716
#define b				2/3   

#endif
