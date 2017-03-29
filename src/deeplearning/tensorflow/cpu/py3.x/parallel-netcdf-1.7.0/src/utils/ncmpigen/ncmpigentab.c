#ifndef lint
static const char yysccsid[] = "@(#)yaccpar	1.9 (Berkeley) 02/21/93";
#endif

#include <stdlib.h>
#include <string.h>

#define YYBYACC 1
#define YYMAJOR 1
#define YYMINOR 9
#define YYPATCH 20070509

#define YYEMPTY (-1)
#define yyclearin    (yychar = YYEMPTY)
#define yyerrok      (yyerrflag = 0)
#define YYRECOVERING (yyerrflag != 0)

extern int yyparse(void);

static int yygrowstack(void);
#define yyparse ncmpiparse
#define yylex ncmpilex
#define yyerror ncmpierror
#define yychar ncmpichar
#define yyval ncmpival
#define yylval ncmpilval
#define yydebug ncmpidebug
#define yynerrs ncmpinerrs
#define yyerrflag ncmpierrflag
#define yyss ncmpiss
#define yyssp ncmpissp
#define yyvs ncmpivs
#define yyvsp ncmpivsp
#define yylhs ncmpilhs
#define yylen ncmpilen
#define yydefred ncmpidefred
#define yydgoto ncmpidgoto
#define yysindex ncmpisindex
#define yyrindex ncmpirindex
#define yygindex ncmpigindex
#define yytable ncmpitable
#define yycheck ncmpicheck
#define yyname ncmpiname
#define yyrule ncmpirule
#define YYPREFIX "ncmpi"
#line 10 "./ncmpigen.y"
#ifdef sccs
static char SccsId[] = "$Id: ncmpigentab.c 2242 2015-12-19 00:37:52Z wkliao $";
#endif

#include        <string.h>
#include	<stdlib.h>
#include	<stddef.h>  /* ptrdiff_t */
#include	<pnetcdf.h>
#include 	"generic.h"
#include        "ncmpigen.h"
#include	"genlib.h"	/* for grow_darray() et al */

typedef struct Symbol {		/* symbol table entry */
	char    	*name;
	struct Symbol   *next;
	unsigned	is_dim : 1;	/* appears as netCDF dimension */
	unsigned	is_var : 1;	/* appears as netCDF variable */
	unsigned	is_att : 1;	/* appears as netCDF attribute */
	int             dnum;	        /* handle as a dimension */
	int             vnum;	        /* handle as a variable */
	} *YYSTYPE1;

/* True if string a equals string b*/
#define	STREQ(a, b)	(*(a) == *(b) && strcmp((a), (b)) == 0)
#define NC_UNSPECIFIED ((nc_type)0)	/* unspecified (as yet) type */

#define YYSTYPE YYSTYPE1
YYSTYPE symlist;		/* symbol table: linked list */

extern int derror_count;	/* counts errors in netcdf definition */
extern int lineno;		/* line number for error messages */

static int not_a_string;	/* whether last constant read was a string */
static char termstring[MAXTRST]; /* last terminal string read */
static double double_val;	/* last double value read */
static float float_val;		/* last float value read */
static int int_val;		/* last int value read */
static short short_val;		/* last short value read */
static char char_val;		/* last char value read */
static signed char byte_val;	/* last byte value read */
static unsigned char ubyte_val;		/* last byte value read */
static unsigned short ushort_val;	/* last byte value read */
static unsigned int uint_val;		/* last byte value read */
static long long int64_val;		/* last byte value read */
static unsigned long long uint64_val;	/* last byte value read */

static nc_type type_code;	/* holds declared type for variables */
static nc_type atype_code;	/* holds derived type for attributes */
static char *netcdfname;	/* to construct netcdf file name */
static void *att_space;		/* pointer to block for attribute values */
static nc_type valtype;		/* type code for list of attribute values  */

static char *char_valp;		/* pointers used to accumulate data values */
static signed char *byte_valp;
static short *short_valp;
static int *int_valp;
static float *float_valp;
static double *double_valp;
static unsigned char *ubyte_valp;
static unsigned short *ushort_valp;
static unsigned int *uint_valp;
static long long *int64_valp;
static unsigned long long *uint64_valp;

static void *rec_cur;		/* pointer to where next data value goes */
static void *rec_start;		/* start of space for data */

/* Forward declarations */
void defatt(void);
void equalatt(void);

#ifdef YYLEX_PARAM
int yylex(YYLEX_PARAM);
#else
int yylex(void);
#endif

#ifdef vms
void yyerror(char*);
#else
int yyerror(char*);
#endif
#line 130 "y.tab.c"
#define NC_UNLIMITED_K 257
#define BYTE_K 258
#define CHAR_K 259
#define SHORT_K 260
#define INT_K 261
#define FLOAT_K 262
#define DOUBLE_K 263
#define UBYTE_K 264
#define USHORT_K 265
#define UINT_K 266
#define INT64_K 267
#define UINT64_K 268
#define IDENT 269
#define TERMSTRING 270
#define BYTE_CONST 271
#define CHAR_CONST 272
#define SHORT_CONST 273
#define INT_CONST 274
#define FLOAT_CONST 275
#define DOUBLE_CONST 276
#define UBYTE_CONST 277
#define USHORT_CONST 278
#define UINT_CONST 279
#define INT64_CONST 280
#define UINT64_CONST 281
#define DIMENSIONS 282
#define VARIABLES 283
#define NETCDF 284
#define DATA 285
#define FILLVALUE 286
#define YYERRCODE 256
short ncmpilhs[] = {                                        -1,
    2,    5,    0,    1,    1,    6,    6,    7,    7,    8,
    8,    8,    9,   10,    3,    3,    3,   11,   11,   13,
   13,   13,   12,   12,   14,   17,   17,   17,   17,   17,
   17,   17,   17,   17,   17,   17,   18,   18,   22,   19,
   20,   21,   21,   23,   23,   24,   26,   15,   29,   16,
   25,   28,   30,   31,   27,   27,   32,   33,   33,   33,
   33,   33,   33,   33,   33,   33,   33,   33,   33,    4,
    4,    4,   34,   34,   36,   35,   37,   37,   40,   38,
   39,   39,   39,   39,   39,   39,   39,   39,   39,   39,
   39,   39,   39,
};
short ncmpilen[] = {                                         2,
    0,    0,    8,    0,    2,    2,    3,    1,    3,    3,
    3,    3,    1,    1,    0,    2,    1,    2,    3,    1,
    1,    1,    2,    3,    2,    1,    1,    1,    1,    1,
    1,    1,    1,    1,    1,    1,    1,    3,    0,    3,
    1,    0,    3,    1,    3,    1,    0,    4,    0,    4,
    3,    2,    1,    1,    1,    3,    1,    1,    1,    1,
    1,    1,    1,    1,    1,    1,    1,    1,    1,    0,
    2,    1,    2,    3,    0,    4,    1,    3,    0,    2,
    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
    1,    1,    1,
};
short ncmpidefred[] = {                                      0,
    0,    0,    1,    0,    0,    0,   14,    0,    0,    8,
    0,   13,    0,    0,    2,    0,    0,   49,    0,    6,
    0,    0,   26,   27,   28,   29,   30,   31,   32,   33,
   34,   35,   36,   41,    0,    0,   20,   21,   22,    0,
   53,   47,    0,   54,   52,    0,    0,   23,    0,    7,
    9,   12,   10,   11,    0,   18,    0,   37,   39,    0,
    0,    0,    0,   24,    0,   19,    0,    0,    0,   51,
   75,    0,    0,    3,   59,   60,   58,   61,   62,   63,
   64,   65,   66,   67,   68,   69,    0,   55,   57,   38,
    0,   40,    0,    0,    0,   73,    0,   46,    0,   44,
   79,   74,   56,    0,   43,    0,   77,    0,   45,   79,
   82,   83,   81,   84,   85,   86,   87,   88,   89,   90,
   91,   92,   93,   80,   78,
};
short ncmpidgoto[] = {                                       2,
    6,    4,   15,   63,   46,    8,    9,   10,   11,   12,
   35,   16,   36,   37,   38,   39,   40,   57,   58,   41,
   92,   68,   99,  100,   42,   60,   87,   18,   49,   43,
   45,   88,   89,   72,   73,   94,  106,  107,  124,  108,
};
short ncmpisindex[] = {                                   -253,
  -90,    0,    0, -246, -215,  -54,    0, -215,  -35,    0,
   -6,    0,  -56, -213,    0,   -1,    1,    0,  -22,    0,
 -215, -249,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,  -56,    2,    0,    0,    0, -211,
    0,    0,    4,    0,    0, -226,    5,    0,    8,    0,
    0,    0,    0,    0,    6,    0,   19,    0,    0,    9,
 -213, -211,  -59,    0, -228,    0, -211,   31, -228,    0,
    0, -211,   13,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,   29,    0,    0,    0,
 -215,    0,   29,   14,   15,    0, -228,    0,   -9,    0,
    0,    0,    0, -215,    0,   32,    0, -260,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,
};
short ncmpirindex[] = {                                      0,
    0,    0,    0,  -58,    0, -122,    0,  -57,    0,    0,
    0,    0,    0,    0,    0, -120,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0, -119,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,  -48,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,   20,    0,    0,    0,
    0,  -47,    0,    0,    0,    0,    0,  -21,    0,    0,
    0,  -45,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,   22,    0,    0,    0,
    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,   24,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,
};
short ncmpigindex[] = {                                      0,
    0,    0,    0,    0,    0,    0,   74,   63,    0,  -63,
    0,    0,   50,    0,    0,   23,    0,    0,   21,  -33,
    0,    0,    0,  -18,    0,    0,   18,    0,    0,  -32,
   28,   -7,    0,    0,   25,    0,    0,  -19,    0,    0,
};
#define YYTABLESIZE 229
short ncmpitable[] = {                                       4,
    5,   14,   15,   14,   17,   16,   59,   52,   21,  111,
  112,  113,  114,  115,  116,  117,  118,  119,  120,  121,
  122,   21,   42,   20,   53,  123,   54,   98,   17,   71,
    1,  105,    3,   59,  104,    5,   50,   42,   47,   71,
   98,   75,   76,   77,   78,   79,   80,   81,   82,   83,
   84,   85,   86,    7,   22,   44,   14,   34,   62,   48,
   56,   61,   67,   64,   66,   74,    4,    5,   65,   69,
   91,   96,   97,  102,  101,  110,   70,   72,   25,   71,
   50,   19,   76,   51,   55,  109,   93,   90,   70,  103,
  125,    0,    0,    0,    0,    0,   95,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,   15,    0,   17,   16,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,   23,   24,   25,   26,   27,   28,   29,   30,   31,
   32,   33,   34,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    4,    5,    4,    5,   13,
};
short ncmpicheck[] = {                                      58,
   58,   58,  125,   58,  125,  125,   40,  257,   44,  270,
  271,  272,  273,  274,  275,  276,  277,  278,  279,  280,
  281,   44,   44,   59,  274,  286,  276,   91,    6,   62,
  284,   41,  123,   67,   44,  282,   59,   59,   16,   72,
  104,  270,  271,  272,  273,  274,  275,  276,  277,  278,
  279,  280,  281,  269,   61,  269,   58,  269,  285,   59,
   59,   58,   44,   59,   59,  125,  125,  125,   61,   61,
   40,   59,   44,   59,   61,   44,  125,  125,   59,  125,
   59,    8,   59,   21,   35,  104,   69,   67,   61,   97,
  110,   -1,   -1,   -1,   -1,   -1,   72,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,  285,   -1,  285,  285,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,  258,  259,  260,  261,  262,  263,  264,  265,  266,
  267,  268,  269,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,  283,  283,  285,  285,  283,
};
#define YYFINAL 2
#ifndef YYDEBUG
#define YYDEBUG 0
#endif
#define YYMAXTOKEN 286
#if YYDEBUG
char *ncmpiname[] = {
"end-of-file",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,"'('","')'",0,0,"','",0,0,0,0,0,0,0,0,0,0,0,0,0,"':'","';'",0,"'='",
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"'{'",0,"'}'",0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
"NC_UNLIMITED_K","BYTE_K","CHAR_K","SHORT_K","INT_K","FLOAT_K","DOUBLE_K",
"UBYTE_K","USHORT_K","UINT_K","INT64_K","UINT64_K","IDENT","TERMSTRING",
"BYTE_CONST","CHAR_CONST","SHORT_CONST","INT_CONST","FLOAT_CONST",
"DOUBLE_CONST","UBYTE_CONST","USHORT_CONST","UINT_CONST","INT64_CONST",
"UINT64_CONST","DIMENSIONS","VARIABLES","NETCDF","DATA","FILLVALUE",
};
char *ncmpirule[] = {
"$accept : ncdesc",
"$$1 :",
"$$2 :",
"ncdesc : NETCDF '{' $$1 dimsection vasection $$2 datasection '}'",
"dimsection :",
"dimsection : DIMENSIONS dimdecls",
"dimdecls : dimdecline ';'",
"dimdecls : dimdecls dimdecline ';'",
"dimdecline : dimdecl",
"dimdecline : dimdecline ',' dimdecl",
"dimdecl : dimd '=' INT_CONST",
"dimdecl : dimd '=' DOUBLE_CONST",
"dimdecl : dimd '=' NC_UNLIMITED_K",
"dimd : dim",
"dim : IDENT",
"vasection :",
"vasection : VARIABLES vadecls",
"vasection : gattdecls",
"vadecls : vadecl ';'",
"vadecls : vadecls vadecl ';'",
"vadecl : vardecl",
"vadecl : attdecl",
"vadecl : gattdecl",
"gattdecls : gattdecl ';'",
"gattdecls : gattdecls gattdecl ';'",
"vardecl : type varlist",
"type : BYTE_K",
"type : CHAR_K",
"type : SHORT_K",
"type : INT_K",
"type : FLOAT_K",
"type : DOUBLE_K",
"type : UBYTE_K",
"type : USHORT_K",
"type : UINT_K",
"type : INT64_K",
"type : UINT64_K",
"varlist : varspec",
"varlist : varlist ',' varspec",
"$$3 :",
"varspec : var $$3 dimspec",
"var : IDENT",
"dimspec :",
"dimspec : '(' dimlist ')'",
"dimlist : vdim",
"dimlist : dimlist ',' vdim",
"vdim : dim",
"$$4 :",
"attdecl : att $$4 '=' attvallist",
"$$5 :",
"gattdecl : gatt $$5 '=' attvallist",
"att : avar ':' attr",
"gatt : ':' attr",
"avar : var",
"attr : IDENT",
"attvallist : aconst",
"attvallist : attvallist ',' aconst",
"aconst : attconst",
"attconst : CHAR_CONST",
"attconst : TERMSTRING",
"attconst : BYTE_CONST",
"attconst : SHORT_CONST",
"attconst : INT_CONST",
"attconst : FLOAT_CONST",
"attconst : DOUBLE_CONST",
"attconst : UBYTE_CONST",
"attconst : USHORT_CONST",
"attconst : UINT_CONST",
"attconst : INT64_CONST",
"attconst : UINT64_CONST",
"datasection :",
"datasection : DATA datadecls",
"datasection : DATA",
"datadecls : datadecl ';'",
"datadecls : datadecls datadecl ';'",
"$$6 :",
"datadecl : avar $$6 '=' constlist",
"constlist : dconst",
"constlist : constlist ',' dconst",
"$$7 :",
"dconst : $$7 const",
"const : CHAR_CONST",
"const : TERMSTRING",
"const : BYTE_CONST",
"const : SHORT_CONST",
"const : INT_CONST",
"const : FLOAT_CONST",
"const : DOUBLE_CONST",
"const : UBYTE_CONST",
"const : USHORT_CONST",
"const : UINT_CONST",
"const : INT64_CONST",
"const : UINT64_CONST",
"const : FILLVALUE",
};
#endif
#ifndef YYSTYPE
typedef int YYSTYPE;
#endif
#if YYDEBUG
#include <stdio.h>
#endif

/* define the initial stack-sizes */
#ifdef YYSTACKSIZE
#undef YYMAXDEPTH
#define YYMAXDEPTH  YYSTACKSIZE
#else
#ifdef YYMAXDEPTH
#define YYSTACKSIZE YYMAXDEPTH
#else
#define YYSTACKSIZE 10000
#define YYMAXDEPTH  10000
#endif
#endif

#define YYINITSTACKSIZE 500

int      yydebug;
int      yynerrs;
int      yyerrflag;
int      yychar;
short   *yyssp;
YYSTYPE *yyvsp;
YYSTYPE  yyval;
YYSTYPE  yylval;

/* variables for the parser stack */
static short   *yyss;
static short   *yysslim;
static YYSTYPE *yyvs;
static int      yystacksize;
#line 1185 "./ncmpigen.y"

/* HELPER PROGRAMS */
void defatt(void)
{
    valnum = 0;
    valtype = NC_UNSPECIFIED;
    /* get a large block for attributes, realloc later */
    att_space = emalloc(MAX_NC_ATTSIZE);
    /* make all kinds of pointers point to it */
    char_valp = (char *) att_space;
    byte_valp = (signed char *) att_space;
    short_valp = (short *) att_space;
    int_valp = (int *) att_space;
    float_valp = (float *) att_space;
    double_valp = (double *) att_space;
    ubyte_valp = (unsigned char *) att_space;
    ushort_valp = (unsigned short *) att_space;
     uint_valp = (unsigned int *) att_space;
     int64_valp = (long long *) att_space;
     uint64_valp = (unsigned long long *) att_space;
}

void equalatt(void)
{
    /* check if duplicate attribute for this var */
    int i;
    for(i=0; i<natts; i++) { /* expensive */
        if(atts[i].var == varnum &&
           STREQ(atts[i].name,atts[natts].name)) {
            derror("duplicate attribute %s:%s",
                   vars[varnum].name,atts[natts].name);
        }
    }
    atts[natts].var = varnum ;
    atts[natts].type = valtype;
    atts[natts].len = valnum;
    /* shrink space down to what was really needed */
    att_space = erealloc(att_space, valnum*nctypesize(valtype));
    atts[natts].val = att_space;
    if (STREQ(atts[natts].name, _FillValue) &&
        atts[natts].var != NC_GLOBAL) {
        nc_putfill(atts[natts].type,atts[natts].val,
                   &vars[atts[natts].var].fill_value);
        if(atts[natts].type != vars[atts[natts].var].type) {
            derror("variable %s: %s type mismatch",
                   vars[atts[natts].var].name, _FillValue);
        }
    }
    natts++;
}
/* PROGRAMS */

#ifdef vms
void
#else
int
#endif
yyerror(	/* called for yacc syntax error */
     char *s)
{
	derror(s);
#ifndef vms
	return -1;
#endif
}

/* undefine yywrap macro, in case we are using bison instead of yacc */
#ifdef ncmpiwrap
#undef ncmpiwrap
#endif

int
ncmpiwrap(void)			/* returns 1 on EOF if no more input */
{
    return  1;
}


/* Symbol table operations for ncmpigen tool */

/* Find CDL name in symbol table (linear search).  Note, this has a
 * side-effect: it handles escape characters in the name, deleting
 * single escape characters from the CDL name, before looking it up.
 */
YYSTYPE lookup(char *sname)
{
    YYSTYPE sp;
    deescapify(sname);		/* delete escape chars from names,
				 * e.g. 'ab\:cd\ ef' becomes 
				 * 'ab:cd ef' */
    for (sp = symlist; sp != (YYSTYPE) 0; sp = sp -> next)
	if (STREQ(sp -> name, sname)) {
	    return sp;
	}
    return 0;			/* 0 ==> not found */
}

YYSTYPE install(  /* install sname in symbol table */
	const char *sname)
{
    YYSTYPE sp;

    sp = (YYSTYPE) emalloc (sizeof (struct Symbol));
    sp -> name = (char *) emalloc (strlen (sname) + 1);/* +1 for '\0' */
    (void) strcpy (sp -> name, sname);
    sp -> next = symlist;	/* put at front of list */
    sp -> is_dim = 0;
    sp -> is_var = 0;
    sp -> is_att = 0;
    symlist = sp;
    return sp;
}

void
clearout(void)	/* reset symbol table to empty */
{
    YYSTYPE sp, tp;
    for (sp = symlist; sp != (YYSTYPE) 0;) {
	tp = sp -> next;
	free (sp -> name);
	free ((char *) sp);
	sp = tp;
    }
    symlist = 0;
}

/* get lexical input routine generated by lex  */

/* Keep compile quiet */
#define YY_NO_UNPUT
#define YY_NO_INPUT

#include "ncmpigenyy.c"
#line 581 "y.tab.c"
/* allocate initial stack or double stack size, up to YYMAXDEPTH */
static int yygrowstack(void)
{
    int newsize, i;
    short *newss;
    YYSTYPE *newvs;

    if ((newsize = yystacksize) == 0)
        newsize = YYINITSTACKSIZE;
    else if (newsize >= YYMAXDEPTH)
        return -1;
    else if ((newsize *= 2) > YYMAXDEPTH)
        newsize = YYMAXDEPTH;

    i = yyssp - yyss;
    newss = (yyss != 0)
          ? (short *)realloc(yyss, newsize * sizeof(*newss))
          : (short *)malloc(newsize * sizeof(*newss));
    if (newss == 0)
        return -1;

    yyss  = newss;
    yyssp = newss + i;
    newvs = (yyvs != 0)
          ? (YYSTYPE *)realloc(yyvs, newsize * sizeof(*newvs))
          : (YYSTYPE *)malloc(newsize * sizeof(*newvs));
    if (newvs == 0)
        return -1;

    yyvs = newvs;
    yyvsp = newvs + i;
    yystacksize = newsize;
    yysslim = yyss + newsize - 1;
    return 0;
}

#define YYABORT goto yyabort
#define YYREJECT goto yyabort
#define YYACCEPT goto yyaccept
#define YYERROR goto yyerrlab
int
yyparse(void)
{
    register int yym, yyn, yystate;
#if YYDEBUG
    register const char *yys;

    if ((yys = getenv("YYDEBUG")) != 0)
    {
        yyn = *yys;
        if (yyn >= '0' && yyn <= '9')
            yydebug = yyn - '0';
    }
#endif

    yynerrs = 0;
    yyerrflag = 0;
    yychar = YYEMPTY;

    if (yyss == NULL && yygrowstack()) goto yyoverflow;
    yyssp = yyss;
    yyvsp = yyvs;
    *yyssp = yystate = 0;

yyloop:
    if ((yyn = yydefred[yystate]) != 0) goto yyreduce;
    if (yychar < 0)
    {
        if ((yychar = yylex()) < 0) yychar = 0;
#if YYDEBUG
        if (yydebug)
        {
            yys = 0;
            if (yychar <= YYMAXTOKEN) yys = yyname[yychar];
            if (!yys) yys = "illegal-symbol";
            printf("%sdebug: state %d, reading %d (%s)\n",
                    YYPREFIX, yystate, yychar, yys);
        }
#endif
    }
    if ((yyn = yysindex[yystate]) && (yyn += yychar) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yychar)
    {
#if YYDEBUG
        if (yydebug)
            printf("%sdebug: state %d, shifting to state %d\n",
                    YYPREFIX, yystate, yytable[yyn]);
#endif
        if (yyssp >= yysslim && yygrowstack())
        {
            goto yyoverflow;
        }
        *++yyssp = yystate = yytable[yyn];
        *++yyvsp = yylval;
        yychar = YYEMPTY;
        if (yyerrflag > 0)  --yyerrflag;
        goto yyloop;
    }
    if ((yyn = yyrindex[yystate]) && (yyn += yychar) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yychar)
    {
        yyn = yytable[yyn];
        goto yyreduce;
    }
    if (yyerrflag) goto yyinrecovery;

    yyerror("syntax error");

#ifdef lint
    goto yyerrlab;
#endif

yyerrlab:
    ++yynerrs;

yyinrecovery:
    if (yyerrflag < 3)
    {
        yyerrflag = 3;
        for (;;)
        {
            if ((yyn = yysindex[*yyssp]) && (yyn += YYERRCODE) >= 0 &&
                    yyn <= YYTABLESIZE && yycheck[yyn] == YYERRCODE)
            {
#if YYDEBUG
                if (yydebug)
                    printf("%sdebug: state %d, error recovery shifting\
 to state %d\n", YYPREFIX, *yyssp, yytable[yyn]);
#endif
                if (yyssp >= yysslim && yygrowstack())
                {
                    goto yyoverflow;
                }
                *++yyssp = yystate = yytable[yyn];
                *++yyvsp = yylval;
                goto yyloop;
            }
            else
            {
#if YYDEBUG
                if (yydebug)
                    printf("%sdebug: error recovery discarding state %d\n",
                            YYPREFIX, *yyssp);
#endif
                if (yyssp <= yyss) goto yyabort;
                --yyssp;
                --yyvsp;
            }
        }
    }
    else
    {
        if (yychar == 0) goto yyabort;
#if YYDEBUG
        if (yydebug)
        {
            yys = 0;
            if (yychar <= YYMAXTOKEN) yys = yyname[yychar];
            if (!yys) yys = "illegal-symbol";
            printf("%sdebug: state %d, error recovery discards token %d (%s)\n",
                    YYPREFIX, yystate, yychar, yys);
        }
#endif
        yychar = YYEMPTY;
        goto yyloop;
    }

yyreduce:
#if YYDEBUG
    if (yydebug)
        printf("%sdebug: state %d, reducing by rule %d (%s)\n",
                YYPREFIX, yystate, yyn, yyrule[yyn]);
#endif
    yym = yylen[yyn];
    if (yym)
        yyval = yyvsp[1-yym];
    else
        memset(&yyval, 0, sizeof yyval);
    switch (yyn)
    {
case 1:
#line 136 "./ncmpigen.y"
{ init_netcdf(); }
break;
case 2:
#line 139 "./ncmpigen.y"
{
		       if (derror_count == 0)
			 define_netcdf(netcdfname);
		       if (derror_count > 0)
			   exit(6);
		   }
break;
case 3:
#line 147 "./ncmpigen.y"
{
		       if (derror_count == 0)
			 close_netcdf();
		   }
break;
case 10:
#line 162 "./ncmpigen.y"
{ if (int_val <= 0)
			 derror("dimension length must be positive");
		     dims[ndims].size = int_val;
		     ndims++;
		   }
break;
case 11:
#line 168 "./ncmpigen.y"
{ /* for rare case where 2^31 < dimsize < 2^32 */
		       if (double_val <= 0)
			 derror("dimension length must be positive");
		       if (double_val > 4294967295.0)
			 derror("dimension too large");
		       if (double_val - (MPI_Offset) double_val > 0)
			 derror("dimension length must be an integer");
		       dims[ndims].size = (MPI_Offset) double_val;
		       ndims++;
                   }
break;
case 12:
#line 179 "./ncmpigen.y"
{  if (rec_dim != -1)
			 derror("only one NC_UNLIMITED dimension allowed");
		     rec_dim = ndims; /* the unlimited (record) dimension */
		     dims[ndims].size = NC_UNLIMITED;
		     ndims++;
		   }
break;
case 13:
#line 187 "./ncmpigen.y"
{ 
		    if (yyvsp[0]->is_dim == 1) {
		        derror( "duplicate dimension declaration for %s",
		                yyvsp[0]->name);
		     }
	             yyvsp[0]->is_dim = 1;
		     yyvsp[0]->dnum = ndims;
		     /* make sure dims array will hold dimensions */
		     grow_darray(ndims,  /* must hold ndims+1 dims */
				 &dims); /* grow as needed */
		     dims[ndims].name = (char *) emalloc(strlen(yyvsp[0]->name)+1);
		     (void) strcpy(dims[ndims].name, yyvsp[0]->name);
		     /* name for use in generated Fortran and C variables */
		     dims[ndims].lname = decodify(yyvsp[0]->name);
		   }
break;
case 26:
#line 219 "./ncmpigen.y"
{ type_code = NC_BYTE; }
break;
case 27:
#line 220 "./ncmpigen.y"
{ type_code = NC_CHAR; }
break;
case 28:
#line 221 "./ncmpigen.y"
{ type_code = NC_SHORT; }
break;
case 29:
#line 222 "./ncmpigen.y"
{ type_code = NC_INT; }
break;
case 30:
#line 223 "./ncmpigen.y"
{ type_code = NC_FLOAT; }
break;
case 31:
#line 224 "./ncmpigen.y"
{ type_code = NC_DOUBLE; }
break;
case 32:
#line 225 "./ncmpigen.y"
{ type_code = NC_UBYTE; }
break;
case 33:
#line 226 "./ncmpigen.y"
{ type_code = NC_USHORT; }
break;
case 34:
#line 227 "./ncmpigen.y"
{ type_code = NC_UINT; }
break;
case 35:
#line 228 "./ncmpigen.y"
{ type_code = NC_INT64; }
break;
case 36:
#line 229 "./ncmpigen.y"
{ type_code = NC_UINT64; }
break;
case 39:
#line 235 "./ncmpigen.y"
{
		    static struct vars dummyvar;

		    dummyvar.name = "dummy";
		    dummyvar.type = NC_DOUBLE;
		    dummyvar.ndims = 0;
		    dummyvar.dims = 0;
		    dummyvar.fill_value.doublev = NC_FILL_DOUBLE;
		    dummyvar.has_data = 0;

		    nvdims = 0;
		    /* make sure variable not re-declared */
		    if (yyvsp[0]->is_var == 1) {
		       derror( "duplicate variable declaration for %s",
		               yyvsp[0]->name);
		    }
	            yyvsp[0]->is_var = 1;
		    yyvsp[0]->vnum = nvars;
		    /* make sure vars array will hold variables */
		    grow_varray(nvars,  /* must hold nvars+1 vars */
				&vars); /* grow as needed */
		    vars[nvars] = dummyvar; /* to make Purify happy */
		    vars[nvars].name = (char *) emalloc(strlen(yyvsp[0]->name)+1);
		    (void) strcpy(vars[nvars].name, yyvsp[0]->name);
		    /* name for use in generated Fortran and C variables */
		    vars[nvars].lname = decodify(yyvsp[0]->name);
		    vars[nvars].type = type_code;
		    /* set default fill value.  You can override this with
		     * the variable attribute "_FillValue". */
		    nc_getfill(type_code, &vars[nvars].fill_value);
		    vars[nvars].has_data = 0; /* has no data (yet) */
		   }
break;
case 40:
#line 268 "./ncmpigen.y"
{
		    vars[nvars].ndims = nvdims;
		    nvars++;
		   }
break;
case 46:
#line 282 "./ncmpigen.y"
{
		    if (nvdims >= NC_MAX_VAR_DIMS) {
		       derror("%s has too many dimensions",vars[nvars].name);
		    }
		    if (yyvsp[0]->is_dim == 1)
		       dimnum = yyvsp[0]->dnum;
		    else {
		       derror( "%s is not declared as a dimension",
			       yyvsp[0]->name);
	               dimnum = ndims;
		    }
		    if (rec_dim != -1 && dimnum == rec_dim && nvdims != 0) {
		       derror("unlimited dimension must be first");
		    }
		    grow_iarray(nvdims, /* must hold nvdims+1 ints */
				&vars[nvars].dims); /* grow as needed */
		    vars[nvars].dims[nvdims] = dimnum;
                    nvdims++;
		   }
break;
case 47:
#line 303 "./ncmpigen.y"
{
                   defatt();
		   }
break;
case 48:
#line 307 "./ncmpigen.y"
{
                   equalatt();
		   }
break;
case 49:
#line 312 "./ncmpigen.y"
{
                   defatt();
		   }
break;
case 50:
#line 316 "./ncmpigen.y"
{
                   equalatt();
		   }
break;
case 52:
#line 324 "./ncmpigen.y"
{
		    varnum = NC_GLOBAL;  /* handle of "global" attribute */
		   }
break;
case 53:
#line 330 "./ncmpigen.y"
{ if (yyvsp[0]->is_var == 1)
		       varnum = yyvsp[0]->vnum;
		    else {
		      derror("%s not declared as a variable, fatal error",
			     yyvsp[0]->name);
		      YYABORT;
		      }
		   }
break;
case 54:
#line 340 "./ncmpigen.y"
{
		       /* make sure atts array will hold attributes */
		       grow_aarray(natts,  /* must hold natts+1 atts */
				   &atts); /* grow as needed */
		       atts[natts].name = (char *) emalloc(strlen(yyvsp[0]->name)+1);
		       (void) strcpy(atts[natts].name,yyvsp[0]->name);
		       /* name for use in generated Fortran and C variables */
		       atts[natts].lname = decodify(yyvsp[0]->name);
		   }
break;
case 57:
#line 354 "./ncmpigen.y"
{
		    if (valtype == NC_UNSPECIFIED)
		      valtype = atype_code;
		    if (valtype != atype_code)
		      derror("values for attribute must be all of same type");
		   }
break;
case 58:
#line 363 "./ncmpigen.y"
{
		       atype_code = NC_CHAR;
		       *char_valp++ = char_val;
		       valnum++;
		   }
break;
case 59:
#line 369 "./ncmpigen.y"
{
		       atype_code = NC_CHAR;
		       {
			   /* don't null-terminate attribute strings */
			   MPI_Offset len = strlen(termstring);
			   if (len == 0) /* need null if that's only value */
			       len = 1;
			   (void)strncpy(char_valp,termstring,len);
			   valnum += len;
			   char_valp += len;
		       }
		   }
break;
case 60:
#line 382 "./ncmpigen.y"
{
		       atype_code = NC_BYTE;
		       *byte_valp++ = byte_val;
		       valnum++;
		   }
break;
case 61:
#line 388 "./ncmpigen.y"
{
		       atype_code = NC_SHORT;
		       *short_valp++ = short_val;
		       valnum++;
		   }
break;
case 62:
#line 394 "./ncmpigen.y"
{
		       atype_code = NC_INT;
		       *int_valp++ = int_val;
		       valnum++;
		   }
break;
case 63:
#line 400 "./ncmpigen.y"
{
		       atype_code = NC_FLOAT;
		       *float_valp++ = float_val;
		       valnum++;
		   }
break;
case 64:
#line 406 "./ncmpigen.y"
{
		       atype_code = NC_DOUBLE;
		       *double_valp++ = double_val;
		       valnum++;
		   }
break;
case 65:
#line 412 "./ncmpigen.y"
{
		       atype_code = NC_UBYTE;
		       *ubyte_valp++ = ubyte_val;
		       valnum++;
		   }
break;
case 66:
#line 418 "./ncmpigen.y"
{
		       atype_code = NC_USHORT;
		       *ushort_valp++ = ushort_val;
		       valnum++;
		   }
break;
case 67:
#line 424 "./ncmpigen.y"
{
		       atype_code = NC_UINT;
		       *uint_valp++ = uint_val;
		       valnum++;
		   }
break;
case 68:
#line 430 "./ncmpigen.y"
{
		       atype_code = NC_INT64;
		       *int64_valp++ = int64_val;
		       valnum++;
		   }
break;
case 69:
#line 436 "./ncmpigen.y"
{
		       atype_code = NC_UINT64;
		       *uint64_valp++ = uint64_val;
		       valnum++;
		   }
break;
case 75:
#line 452 "./ncmpigen.y"
{
		       valtype = vars[varnum].type; /* variable type */
		       valnum = 0;	/* values accumulated for variable */
		       vars[varnum].has_data = 1;
		       /* compute dimensions product */
		       var_size = nctypesize(valtype);
		       if (vars[varnum].ndims == 0) { /* scalar */
			   var_len = 1;
		       } else if (vars[varnum].dims[0] == rec_dim) {
			   var_len = 1; /* one record for unlimited vars */
		       } else {
			   var_len = dims[vars[varnum].dims[0]].size;
		       }
		       for(dimnum = 1; dimnum < vars[varnum].ndims; dimnum++)
			 var_len = var_len*dims[vars[varnum].dims[dimnum]].size;
		       /* allocate memory for variable data */
		       if (var_len*var_size != (MPI_Offset)(var_len*var_size)) {
			   derror("variable %s too large for memory",
				  vars[varnum].name);
			   exit(9);
		       }
		       rec_len = var_len;
		       rec_start = malloc ((MPI_Offset)(rec_len*var_size));
		       if (rec_start == 0) {
			   derror ("out of memory\n");
			   exit(3);
		       }
		       rec_cur = rec_start;
		       switch (valtype) {
			 case NC_CHAR:
			   char_valp = (char *) rec_start;
			   break;
			 case NC_BYTE:
			   byte_valp = (signed char *) rec_start;
			   break;
			 case NC_SHORT:
			   short_valp = (short *) rec_start;
			   break;
			 case NC_INT:
			   int_valp = (int *) rec_start;
			   break;
			 case NC_FLOAT:
			   float_valp = (float *) rec_start;
			   break;
			 case NC_DOUBLE:
			   double_valp = (double *) rec_start;
			   break;
			 case NC_UBYTE:
			   ubyte_valp = (unsigned char *) rec_start;
			   break;
			 case NC_USHORT:
			   ushort_valp = (unsigned short *) rec_start;
			   break;
			 case NC_UINT:
			   uint_valp = (unsigned int *) rec_start;
			   break;
			 case NC_INT64:
			   int64_valp = (long long *) rec_start;
			   break;
			 case NC_UINT64:
			   uint64_valp = (unsigned long long *) rec_start;
			   break;
			 default: break;
		       }
		 }
break;
case 76:
#line 518 "./ncmpigen.y"
{
		       if (valnum < var_len) { /* leftovers */
			   nc_fill(valtype,
				    var_len - valnum,
				    rec_cur,
				    vars[varnum].fill_value);
		       }
		       /* put out var_len values */
		       /* vars[varnum].nrecs = valnum / rec_len; */
		       vars[varnum].nrecs = var_len / rec_len;
		       if (derror_count == 0)
			   put_variable(rec_start);
		       free ((char *) rec_start);
		 }
break;
case 79:
#line 537 "./ncmpigen.y"
{
		       if(valnum >= var_len) {
			   if (vars[varnum].dims[0] != rec_dim) { /* not recvar */
			       derror("too many values for this variable, %d >= %d",
				      valnum, var_len);
			       exit (4);
			   } else { /* a record variable, so grow data
				      container and increment var_len by
				      multiple of record size */
			       ptrdiff_t rec_inc = (char *)rec_cur
				   - (char *)rec_start;
			       var_len = rec_len * (1 + valnum / rec_len);
			       rec_start = erealloc(rec_start, var_len*var_size);
			       rec_cur = (char *)rec_start + rec_inc;
			       char_valp = (char *) rec_cur;
			       byte_valp = (signed char *) rec_cur;
			       short_valp = (short *) rec_cur;
			       int_valp = (int *) rec_cur;
			       float_valp = (float *) rec_cur;
			       double_valp = (double *) rec_cur;
			       ubyte_valp = (unsigned char *) rec_cur;
			       ushort_valp = (unsigned short *) rec_cur;
			       uint_valp = (unsigned int *) rec_cur;
			       int64_valp = (long long *) rec_cur;
			       uint64_valp = (unsigned long long *) rec_cur;
			   }
		       }
		       not_a_string = 1;
                   }
break;
case 80:
#line 567 "./ncmpigen.y"
{
		       if (not_a_string) {
			   switch (valtype) {
			     case NC_CHAR:
			       rec_cur = (void *) char_valp;
			       break;
			     case NC_BYTE:
			       rec_cur = (void *) byte_valp;
			       break;
			     case NC_SHORT:
			       rec_cur = (void *) short_valp;
			       break;
			     case NC_INT:
			       rec_cur = (void *) int_valp;
			       break;
			     case NC_FLOAT:
			       rec_cur = (void *) float_valp;
			       break;
			     case NC_DOUBLE:
			       rec_cur = (void *) double_valp;
			       break;
			     case NC_UBYTE:
			       rec_cur = (void *) ubyte_valp;
			       break;
			     case NC_USHORT:
			       rec_cur = (void *) ushort_valp;
			       break;
			     case NC_UINT:
			       rec_cur = (void *) uint_valp;
			       break;
			     case NC_INT64:
			       rec_cur = (void *) int64_valp;
			       break;
			     case NC_UINT64:
			       rec_cur = (void *) uint64_valp;
			       break;
			     default: break;
			   }
		       }
		   }
break;
case 81:
#line 610 "./ncmpigen.y"
{
		       atype_code = NC_CHAR;
		       switch (valtype) {
			 case NC_CHAR:
			   *char_valp++ = char_val;
			   break;
			 case NC_BYTE:
			   *byte_valp++ = char_val;
			   break;
			 case NC_SHORT:
			   *short_valp++ = char_val;
			   break;
			 case NC_INT:
			   *int_valp++ = char_val;
			   break;
			 case NC_FLOAT:
			   *float_valp++ = char_val;
			   break;
			 case NC_DOUBLE:
			   *double_valp++ = char_val;
			   break;
			 case NC_UBYTE:
			   *ubyte_valp++ = char_val;
			   break; 
			 case NC_USHORT:
			   *ushort_valp++ = char_val;
			   break; 
			 case NC_UINT:
			   *uint_valp++ = char_val;
			   break; 
			 case NC_INT64:
			   *int64_valp++ = char_val;
			   break; 
			 case NC_UINT64:
			   *uint64_valp++ = char_val;
			   break; 
			 default: break;
		       }
		       valnum++;
		   }
break;
case 82:
#line 651 "./ncmpigen.y"
{
		       not_a_string = 0;
		       atype_code = NC_CHAR;
		       {
			   MPI_Offset len = strlen(termstring);

			   if(valnum + len > var_len) {
			       if (vars[varnum].dims[0] != rec_dim) {
				   derror("too many values for this variable, %d>%d", 
					  valnum+len, var_len);
				   exit (5);
			       } else {/* a record variable so grow it */
				   ptrdiff_t rec_inc = (char *)rec_cur
				       - (char *)rec_start;
				   var_len += rec_len * (len + valnum - var_len)/rec_len;
				   rec_start = erealloc(rec_start, var_len*var_size);
				   rec_cur = (char *)rec_start + rec_inc;
				   char_valp = (char *) rec_cur;
			       }
			   }
			   switch (valtype) {
			     case NC_CHAR:
			       {
				   int ld;
				   MPI_Offset i, sl;
				   (void)strncpy(char_valp,termstring,len);
				   ld = vars[varnum].ndims-1;
				   if (ld > 0) {/* null-fill to size of last dim */
				       sl = dims[vars[varnum].dims[ld]].size;
				       for (i =len;i<sl;i++)
					   char_valp[i] = '\0';
				       if (sl < len)
					   sl = len;
				       valnum += sl;
				       char_valp += sl;
				   } else { /* scalar or 1D strings */
				       valnum += len;
				       char_valp += len;
				   }
				   rec_cur = (void *) char_valp;
			       }
			       break;
			     case NC_BYTE:
			     case NC_SHORT:
			     case NC_INT:
			     case NC_FLOAT:
			     case NC_DOUBLE:
			       derror("string value invalid for %s variable",
				      nctype(valtype));
			       break;
			     default: break;
			   }
		       }
		   }
break;
case 83:
#line 706 "./ncmpigen.y"
{
		       atype_code = NC_BYTE;
		       switch (valtype) {
			 case NC_CHAR:
			   *char_valp++ = byte_val;
			   break;
			 case NC_BYTE:
			   *byte_valp++ = byte_val;
			   break;
			 case NC_SHORT:
			   *short_valp++ = byte_val;
			   break;
			 case NC_INT:
			   *int_valp++ = byte_val;
			   break;
			 case NC_FLOAT:
			   *float_valp++ = byte_val;
			   break;
			 case NC_DOUBLE:
			   *double_valp++ = byte_val;
			   break;
			 case NC_UBYTE:
			   *ubyte_valp++ = byte_val;
			   break;
			 case NC_USHORT:
			   *ushort_valp++ = byte_val;
			   break;
			 case NC_UINT:
			   *uint_valp++ = byte_val;
			   break;
			 case NC_INT64:
			   *int64_valp++ = byte_val;
			   break;
			 case NC_UINT64:
			   *uint64_valp++ = byte_val;
			   break;
			 default: break;
		       }
		       valnum++;
		   }
break;
case 84:
#line 747 "./ncmpigen.y"
{
		       atype_code = NC_SHORT;
		       switch (valtype) {
			 case NC_CHAR:
			   *char_valp++ = short_val;
			   break;
			 case NC_BYTE:
			   *byte_valp++ = short_val;
			   break;
			 case NC_SHORT:
			   *short_valp++ = short_val;
			   break;
			 case NC_INT:
			   *int_valp++ = short_val;
			   break;
			 case NC_FLOAT:
			   *float_valp++ = short_val;
			   break;
			 case NC_DOUBLE:
			   *double_valp++ = short_val;
			   break;
			 case NC_UBYTE:
			   *ubyte_valp++ = short_val;
			   break;
			 case NC_USHORT:
			   *ushort_valp++ = short_val;
			   break;
			 case NC_UINT:
			   *uint_valp++ = short_val;
			   break;
			 case NC_INT64:
			   *int64_valp++ = short_val;
			   break;
			 case NC_UINT64:
			   *uint64_valp++ = short_val;
			   break;
			 default: break;			
		       }
		       valnum++;
		   }
break;
case 85:
#line 788 "./ncmpigen.y"
{
		       atype_code = NC_INT;
		       switch (valtype) {
			 case NC_CHAR:
			   *char_valp++ = int_val;
			   break;
			 case NC_BYTE:
			   *byte_valp++ = int_val;
			   break;
			 case NC_SHORT:
			   *short_valp++ = int_val;
			   break;
			 case NC_INT:
			   *int_valp++ = int_val;
			   break;
			 case NC_FLOAT:
			   *float_valp++ = int_val;
			   break;
			 case NC_DOUBLE:
			   *double_valp++ = int_val;
			   break;
			 case NC_UBYTE:
			   *ubyte_valp++ = int_val;
			   break;
			 case NC_USHORT:
			   *ushort_valp++ = int_val;
			   break;
			 case NC_UINT:
			   *uint_valp++ = int_val;
			   break;
			 case NC_INT64:
			   *int64_valp++ = int_val;
			   break;
			 case NC_UINT64:
			   *uint64_valp++ = int_val;
			   break;
			 default: break;
		       }
		       valnum++;
		   }
break;
case 86:
#line 829 "./ncmpigen.y"
{
		       atype_code = NC_FLOAT;
		       switch (valtype) {
			 case NC_CHAR:
			   *char_valp++ = float_val;
			   break;
			 case NC_BYTE:
			   *byte_valp++ = float_val;
			   break;
			 case NC_SHORT:
			   *short_valp++ = float_val;
			   break;
			 case NC_INT:
			   *int_valp++ = float_val;
			   break;
			 case NC_FLOAT:
			   *float_valp++ = float_val;
			   break;
			 case NC_DOUBLE:
			   *double_valp++ = float_val;
			   break;
			 case NC_UBYTE:
			   *ubyte_valp++ = float_val;
			   break;
			 case NC_USHORT:
			   *ushort_valp++ = float_val;
			   break;
			 case NC_UINT:
			   *uint_valp++ = float_val;
			   break;
			 case NC_INT64:
			   *int64_valp++ = float_val;
			   break;
			 case NC_UINT64:
			   *uint64_valp++ = float_val;
			   break;
			 default: break;
		       }
		       valnum++;
		   }
break;
case 87:
#line 870 "./ncmpigen.y"
{
		       atype_code = NC_DOUBLE;
		       switch (valtype) {
			 case NC_CHAR:
			   *char_valp++ = double_val;
			   break;
			 case NC_BYTE:
			   *byte_valp++ = double_val;
			   break;
			 case NC_SHORT:
			   *short_valp++ = double_val;
			   break;
			 case NC_INT:
			   *int_valp++ = double_val;
			   break;
			 case NC_FLOAT:
			   if (double_val == NC_FILL_DOUBLE)
			     *float_valp++ = NC_FILL_FLOAT;
			   else
			     *float_valp++ = double_val;
			   break;
			 case NC_DOUBLE:
			   *double_valp++ = double_val;
			   break;
			 case NC_UBYTE:
			   *ubyte_valp++ = double_val;
			   break;
			 case NC_USHORT:
			   *ushort_valp++ = double_val;
			   break;
			 case NC_UINT:
			   *uint_valp++ = double_val;
			   break;
			 case NC_INT64:
			   *int64_valp++ = double_val;
			   break;
			 case NC_UINT64:
			   *uint64_valp++ = double_val;
			   break;
			 default: break;
		       }
		       valnum++;
		   }
break;
case 88:
#line 914 "./ncmpigen.y"
{
		       atype_code = NC_UBYTE;
		       switch (valtype) {
			 case NC_CHAR:
			   *char_valp++ = ubyte_val;
			   break;
			 case NC_BYTE:
			   *byte_valp++ = ubyte_val;
			   break;
			 case NC_SHORT:
			   *short_valp++ = ubyte_val;
			   break;
			 case NC_INT:
			   *int_valp++ = ubyte_val;
			   break;
			 case NC_FLOAT:
			     *float_valp++ = ubyte_val;
			   break;
			 case NC_DOUBLE:
			   *double_valp++ = ubyte_val;
			   break;
			 case NC_UBYTE:
			   *ubyte_valp++ = ubyte_val;
			   break;
			 case NC_USHORT:
			   *ushort_valp++ = ubyte_val;
			   break;
			 case NC_UINT:
			   *uint_valp++ = ubyte_val;
			   break;
			 case NC_INT64:
			   *int64_valp++ = ubyte_val;
			   break;
			 case NC_UINT64:
			   *uint64_valp++ = ubyte_val;
			   break;
			 default: 
			   derror("Unhandled type %d\n", valtype);
			   break;
		       }
		       valnum++;
		   }
break;
case 89:
#line 957 "./ncmpigen.y"
{
		       atype_code = NC_USHORT;
		       switch (valtype) {
			 case NC_CHAR:
			   *char_valp++ = ushort_val;
			   break;
			 case NC_BYTE:
			   *byte_valp++ = ushort_val;
			   break;
			 case NC_SHORT:
			   *short_valp++ = ushort_val;
			   break;
			 case NC_INT:
			   *int_valp++ = ushort_val;
			   break;
			 case NC_FLOAT:
			   *float_valp++ = ushort_val;
			   break;
			 case NC_DOUBLE:
			   *double_valp++ = ushort_val;
			   break;
			 case NC_UBYTE:
			   *ubyte_valp++ = ushort_val;
			   break;
			 case NC_USHORT:
			   *ushort_valp++ = ushort_val;
			   break;
			 case NC_UINT:
			   *uint_valp++ = ushort_val;
			   break;
			 case NC_INT64:
			   *int64_valp++ = ushort_val;
			   break;
			 case NC_UINT64:
			   *uint64_valp++ = ushort_val;
			   break;
			 default: 
			   derror("Unhandled type %d\n", valtype);
			   break;
		       }
		       valnum++;
		   }
break;
case 90:
#line 1000 "./ncmpigen.y"
{
		       atype_code = NC_UINT;
		       switch (valtype) {
			 case NC_CHAR:
			   *char_valp++ = uint_val;
			   break;
			 case NC_BYTE:
			   *byte_valp++ = uint_val;
			   break;
			 case NC_SHORT:
			   *short_valp++ = uint_val;
			   break;
			 case NC_INT:
			   *int_valp++ = uint_val;
			   break;
			 case NC_FLOAT:
			   *float_valp++ = uint_val;
			   break;
			 case NC_DOUBLE:
			   *double_valp++ = uint_val;
			   break;
			 case NC_UBYTE:
			   *ubyte_valp++ = uint_val;
			   break;
			 case NC_USHORT:
			   *ushort_valp++ = uint_val;
			   break;
			 case NC_UINT:
			   *uint_valp++ = uint_val;
			   break;
			 case NC_INT64:
			   *int64_valp++ = uint_val;
			   break;
			 case NC_UINT64:
			   *uint64_valp++ = uint_val;
			   break;
			 default: 
			   derror("Unhandled type %d\n", valtype);
			   break;
		       }
		       valnum++;
		   }
break;
case 91:
#line 1043 "./ncmpigen.y"
{
		       atype_code = NC_INT64;
		       switch (valtype) {
			 case NC_CHAR:
			   *char_valp++ = int64_val;
			   break;
			 case NC_BYTE:
			   *byte_valp++ = int64_val;
			   break;
			 case NC_SHORT:
			   *short_valp++ = int64_val;
			   break;
			 case NC_INT:
			   *int_valp++ = int64_val;
			   break;
			 case NC_FLOAT:
			   *float_valp++ = int64_val;
			   break;
			 case NC_DOUBLE:
			   *double_valp++ = int64_val;
			   break;
			 case NC_UBYTE:
			   *ubyte_valp++ = int64_val;
			   break;
			 case NC_USHORT:
			   *ushort_valp++ = int64_val;
			   break;
			 case NC_UINT:
			   *uint_valp++ = int64_val;
			   break;
			 case NC_INT64:
			   *int64_valp++ = int64_val;
			   break;
			 case NC_UINT64:
			   *uint64_valp++ = int64_val;
			   break;
			 default: 
			   derror("Unhandled type %d\n", valtype);
			   break;
		       }
		       valnum++;
		   }
break;
case 92:
#line 1086 "./ncmpigen.y"
{
		       atype_code = NC_UINT64;
		       switch (valtype) {
			 case NC_CHAR:
			   *char_valp++ = uint64_val;
			   break;
			 case NC_BYTE:
			   *byte_valp++ = uint64_val;
			   break;
			 case NC_SHORT:
			   *short_valp++ = uint64_val;
			   break;
			 case NC_INT:
			   *int_valp++ = uint64_val;
			   break;
			 case NC_FLOAT:
			   *float_valp++ = uint64_val;
			   break;
			 case NC_DOUBLE:
			   *double_valp++ = uint64_val;
			   break;
			 case NC_UBYTE:
			   *ubyte_valp++ = uint64_val;
			   break;
			 case NC_USHORT:
			   *ushort_valp++ = uint64_val;
			   break;
			 case NC_UINT:
			   *uint_valp++ = uint64_val;
 			   break;
 			 case NC_INT64:
 			   *int64_valp++ = uint64_val;
 			   break;
 			 case NC_UINT64:
 			   *uint64_valp++ = uint64_val;
 			   break;
 			 default: 
 			   derror("Unhandled type %d\n", valtype);
 			   break;
 		       }
 		       valnum++;
 		   }
break;
case 93:
#line 1129 "./ncmpigen.y"
{
		       /* store fill_value */
		       switch (valtype) {
		       case NC_CHAR:
			   nc_fill(valtype, 1, (void *)char_valp++,
				   vars[varnum].fill_value);
			   break;
		       case NC_BYTE:
			   nc_fill(valtype, 1, (void *)byte_valp++,
				   vars[varnum].fill_value);
			   break;
		       case NC_SHORT:
			   nc_fill(valtype, 1, (void *)short_valp++,
				   vars[varnum].fill_value);
			   break;
		       case NC_INT:
			   nc_fill(valtype, 1, (void *)int_valp++,
				   vars[varnum].fill_value);
			   break;
		       case NC_FLOAT:
			   nc_fill(valtype, 1, (void *)float_valp++,
				   vars[varnum].fill_value);
			   break;
		       case NC_DOUBLE:
			   nc_fill(valtype, 1, (void *)double_valp++,
				   vars[varnum].fill_value);
			   break;
		       case NC_UBYTE:
			   nc_fill(valtype, 1, (void *)ubyte_valp++,
				   vars[varnum].fill_value);
			   break;
		       case NC_USHORT:
			   nc_fill(valtype, 1, (void *)ushort_valp++,
				   vars[varnum].fill_value);
			   break;
		       case NC_UINT:
			   nc_fill(valtype, 1, (void *)uint_valp++,
				   vars[varnum].fill_value);
			   break;
		       case NC_INT64:
			   nc_fill(valtype, 1, (void *)int64_valp++,
				   vars[varnum].fill_value);
			   break;
		       case NC_UINT64:
			   nc_fill(valtype, 1, (void *)uint64_valp++,
				   vars[varnum].fill_value);
			   break;
			default: break;
		       }
		       valnum++;
		   }
break;
#line 1860 "y.tab.c"
    }
    yyssp -= yym;
    yystate = *yyssp;
    yyvsp -= yym;
    yym = yylhs[yyn];
    if (yystate == 0 && yym == 0)
    {
#if YYDEBUG
        if (yydebug)
            printf("%sdebug: after reduction, shifting from state 0 to\
 state %d\n", YYPREFIX, YYFINAL);
#endif
        yystate = YYFINAL;
        *++yyssp = YYFINAL;
        *++yyvsp = yyval;
        if (yychar < 0)
        {
            if ((yychar = yylex()) < 0) yychar = 0;
#if YYDEBUG
            if (yydebug)
            {
                yys = 0;
                if (yychar <= YYMAXTOKEN) yys = yyname[yychar];
                if (!yys) yys = "illegal-symbol";
                printf("%sdebug: state %d, reading %d (%s)\n",
                        YYPREFIX, YYFINAL, yychar, yys);
            }
#endif
        }
        if (yychar == 0) goto yyaccept;
        goto yyloop;
    }
    if ((yyn = yygindex[yym]) && (yyn += yystate) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yystate)
        yystate = yytable[yyn];
    else
        yystate = yydgoto[yym];
#if YYDEBUG
    if (yydebug)
        printf("%sdebug: after reduction, shifting from state %d \
to state %d\n", YYPREFIX, *yyssp, yystate);
#endif
    if (yyssp >= yysslim && yygrowstack())
    {
        goto yyoverflow;
    }
    *++yyssp = yystate;
    *++yyvsp = yyval;
    goto yyloop;

yyoverflow:
    yyerror("yacc stack overflow");

yyabort:
    return (1);

yyaccept:
    return (0);
}

