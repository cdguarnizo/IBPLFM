#if defined(WITH_COMMENTS)
/*
 * fintrf.h	- MATLAB/FORTRAN interface header file. This file
 *		  contains the declaration of the pointer type needed
 *		  by the MATLAB/FORTRAN interface.
 *
 * Copyright (c) 1984-98 by The MathWorks, Inc.
 * All Rights Reserved.
 * $Revision: 1.11 $  $Date: 1999/06/22 15:23:26 $
 */
#endif
#if defined(__alpha) || (defined(__sgi) && _MIPS_SZPTR==64)
#define mwpointer integer*8
#define MWPOINTER INTEGER*8
#else
#define mwpointer integer
#define MWPOINTER INTEGER
#endif