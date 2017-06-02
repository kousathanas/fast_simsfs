#include <math.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_linalg.h>
#include <getopt.h>
#include <string.h>

#include "libhdr"
#include "Library_DFE_v1.8.h"


/*
  version 1.7.3: 
  -added: constant population support
  -added: generate gamma distribution deviates
  -added: generate beta distribution deviates
  -added: generate gamma+beta distribution deviates

  to compile:

  gcc -O3 -o bin/Multi_DFE_gen_v1.8 Multi_DFE_gen_v1.8.c Library_DFE_v1.8.c tmatrix_routines.c genlib.c nrlib.c nrutil.c  -lm -lgsl -lgslcblas -w


  example run:

  GSL_RNG_SEED=1 ~/Multi_DFE_est/source/bin/Multi_DFE_gen_v1.8 -N1 100 -N2 100 -nalleles 20 -t 100 -f0 0.9 -neutral 1000000 -selected 1000000 -mode 9 -conpop 1 -exp_mean 0.05 -file 1.out

  #-gamma_alpha 10 -gamma_beta 0.5
  #-exp_mean 0.02 -beta_alpha 10 -beta_beta 1
  #-exp_mean 0.02 -gamma_alpha 1000 -gamma_beta 50
  #-gamma_alpha 70 -gamma_beta 0.07 -gamma2_alpha 1000 -gamma2_beta 50
  #-exp_mean 0.05

*/


/*Global variable declaration*/
int selmode=0;
int  n_sfs=0;
int nspikes=1;//default a single spike

int N1,N2=100;
int max_n2d=2000;
double saveresults[100];
int n2_step=0, conpop=0, output_egf_mode=0;

/*Function list*/
double load_FV(int n1,int n2,int t2,double s,double f0,double *mean_FV);

/******************************************************************************/
double load_FV(int n1,int n2,int t2,double s,double f0,double *mean_FV)
{
  
  static double egf_vec1_lower[maxnd+1], egf_vec2_lower[maxnd+1], 
    egf_vec1_upper[maxnd+1], egf_vec2_upper[maxnd+1],
    egf_vec1[maxnd+1], egf_vec2[maxnd+1],egf_vec[maxnd+1];

  double *gamma_density_vec;

  int i=0,j=0,file_size_bytes=0;
  int t2_upper=0;
  int t2_real=t2;
  int n1d=2*n1;
  int n2d=2*n2;


  char *buffer_p1_t2_lower, *buffer_p2_t2_lower,
    *buffer_p1_t2_upper, *buffer_p2_t2_upper, *buffer_const_pop;
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if (!conpop)
    {
      get_upper_lower_int(t2_real, &t2_lower, &t2_upper, n_t2_evaluated, t2_evaluated_vec);
      if ((t2_lower==undefined_int)||(t2_upper==undefined_int))
	{
	  return undefined;
	}
      //printf("\n%d",t2_lower);
      //printf("\n%f",s);
      file_size_bytes=compute_file_size_bytes(n2);
      //Read buffers
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      buffer_p1_t2_lower = (char*) malloc (file_size_bytes);
      buffer_p1_t2_upper = (char*) malloc (file_size_bytes);

      buffer_p2_t2_lower = (char*) malloc (file_size_bytes);
      buffer_p2_t2_upper = (char*) malloc (file_size_bytes);

      read_phase1_phase2_file_into_buffer(n1,1, n2, t2_lower,
					  buffer_p1_t2_lower, file_size_bytes);
      read_phase1_phase2_file_into_buffer(n1,1, n2, t2_upper,
					  buffer_p1_t2_upper, file_size_bytes);

      read_phase1_phase2_file_into_buffer(n1,2, n2, t2_lower,
					  buffer_p2_t2_lower, file_size_bytes);
      read_phase1_phase2_file_into_buffer(n1,2, n2, t2_upper,
					  buffer_p2_t2_upper, file_size_bytes);

      get_binary_egf_vec(buffer_p1_t2_lower, n2, t2_lower, s, egf_vec1_lower);
      get_binary_egf_vec(buffer_p1_t2_upper, n2, t2_upper, s, egf_vec1_upper);


      get_binary_egf_vec(buffer_p2_t2_lower, n2, t2_lower, s, egf_vec2_lower);
      get_binary_egf_vec(buffer_p2_t2_upper, n2, t2_upper, s, egf_vec2_upper);
      //End reading buffers
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //Averaging vectors w(s) and x(s)

      compute_weighted_average_egf_vec(t2_real, t2_lower, t2_upper,
				       egf_vec1_lower, egf_vec1_upper, egf_vec1, n2d);
      compute_weighted_average_egf_vec(t2_real, t2_lower, t2_upper,
				       egf_vec2_lower, egf_vec2_upper, egf_vec2, n2d);
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //dumpvector(egf_vec1,0,210,"egf_vec1"); 
      //printf("\n%d\t%d",t2_lower, t2_upper);
      //dumpvector(egf_vec2,0,210,"egf_vec2"); 
      compute_weighted_average( mean_FV, egf_vec1, egf_vec2, n1d, n2d);


      free(buffer_p1_t2_lower);
      free(buffer_p2_t2_lower);
      free(buffer_p1_t2_upper);
      free(buffer_p2_t2_upper);
    }   
  else//constant population
    {
      if (2*n1 > max_n2d)
	{
	  printf("ERROR: Value of 2*n1 %d exceeds max_n2d %d\n", n1, max_n2d);
	  gabort("Program terminating", 0);
	}
      file_size_bytes = compute_file_size_bytes(n1);
      //      printf("const pop: file_size_bytes %d\n", file_size_bytes); monitorinput();
      buffer_const_pop = (char*) malloc (file_size_bytes);

      read_const_pop_file_into_buffer(n1, buffer_const_pop, file_size_bytes);
      get_const_pop_egf_vec(s, n1, mean_FV, buffer_const_pop, 1);


      free(buffer_const_pop);
    } 


  return (1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static int verbose_flag;
main(argc,argv)
int argc; char **argv;
{
  int i=0,j=0;
  int nalleles=0;
  double f0=0;
  int t=0;
  int nspikes=0;
  int sampleS=0,sampleN=0;

  double beta_alpha=0,beta_beta=0,gamma_alpha=0, gamma_beta=0,gamma2_alpha=0,gamma2_beta=0,exp_mean=0,exp_mean2=0,
    sel_coeff1=0,sel_coeff2=0,sel_coeff3=0,prob1=0,prob2=0,prob3=0;
  char *sfs_filename;

  /*
    location parameters measured
    Mean,squared mean,harmonic mean
  */
  double MEAN_S=0,MEAN_S2=0,MEAN_H=0,fix_prob=0;

  int c;
     
  while (1)
    {
      static struct option long_options[] =
	{
	  /* These options set a flag. */
	  // {"verbose", no_argument,       &verbose_flag, 1},
	  // {"brief",   no_argument,       &verbose_flag, 0},
	  /* These options don't set a flag.
	     We distinguish them by their indices. */
	  {"N1",     required_argument,0, 'a'},
	  {"N2",  required_argument,0, 'b'},
	  {"nalleles",  required_argument, 0, 'c'},
	  {"t",  required_argument, 0, 'd'},
	  {"f0",    required_argument, 0, 'e'},
	  {"neutral",    required_argument, 0, 'f'},
	  {"selected",    required_argument, 0, 'g'},
	  {"mode",    required_argument, 0, 'h'},
	  {"nspikes",    required_argument, 0, 'i'},
	  {"gamma_alpha",    required_argument, 0, 'j'},
	  {"gamma_beta",    required_argument, 0, 'k'},
	  {"beta_alpha",    required_argument, 0, 'l'},
	  {"beta_beta",    required_argument, 0, 'm'},
	  {"gamma2_alpha",    required_argument, 0, 'n'},
	  {"gamma2_beta",    required_argument, 0, 'o'},
	  {"exp_mean",    required_argument, 0, 'p'},
	  {"conpop",    required_argument, 0, 'q'},
	  {"file",    required_argument, 0, 'r'},
	  {"exp_mean2",    required_argument, 0, 's'},
	  {"s1",    required_argument, 0, 't'},	  
	  {"s2",    required_argument, 0, 'u'},
	  {"s3",    required_argument, 0, 'v'},	  
	  {"p1",    required_argument, 0, 'w'},
	  {"p2",    required_argument, 0, 'x'},
	  {"p3",    required_argument, 0, 'y'},
	  {0, 0, 0, 0}
	};
      /* getopt_long stores the option index here. */
      int option_index = 0;
     
      c = getopt_long_only (argc, argv, "",
                            long_options, &option_index);
     
      /* Detect the end of the options. */
      if (c == -1)
	break;
     
      switch (c)
	{
	case 0:
	  /* If this option set a flag, do nothing else now. */
	  if (long_options[option_index].flag != 0)
	    break;
	  printf ("option %s", long_options[option_index].name);
	  if (optarg)
	    printf (" with arg %s", optarg);
	  printf ("\nalleles");
	  break;

	case 'a':
	  N1=atoi(optarg);
	  break;   
	case 'b':
	  N2=atoi(optarg);
	  break;     
	case 'c':
	  nalleles=atoi(optarg);
	  break;     
	case 'd':
	  t=atof(optarg);
	  break;     
	case 'e':
	  f0=atof(optarg);
	  break;
	case 'f':
	  sampleS=atoi(optarg);
	  break;
	case 'g':
	  sampleN=atoi(optarg);
	  break;
	case 'h':
	  selmode=atoi(optarg);
	  break;
	case 'i':
	  nspikes=atoi(optarg);
	  break;
	case 'j':
	  gamma_alpha=atof(optarg);
	  break;
	case 'k':
	  gamma_beta=atof(optarg);
	  break;
	case 'l':
	  beta_alpha=atof(optarg);
	  break;
	case 'm':
	  beta_beta=atof(optarg);
	  break;
	case 'n':
	  gamma2_alpha=atof(optarg);
	  break;
	case 'o':
	  gamma2_beta=atof(optarg);
	  break;
	case 'p':
	  exp_mean=atof(optarg);
	  break;
	case 'q':
	  conpop=atoi(optarg);
	  break;
	case 'r':
	  sfs_filename=optarg;
	  break;     
	case 's':
	  exp_mean2=atof(optarg);
	  break;  
	case 't':
	  sel_coeff1=atof(optarg);
	  break; 
	case 'u':
	  sel_coeff2=atof(optarg);
	  break; 
	case 'v':
	  sel_coeff3=atof(optarg);
	  break; 
	case 'w':
	  prob1=atof(optarg);
	  break; 
	case 'x':
	  prob2=atof(optarg);
	  break; 
	case 'y':
	  prob3=atof(optarg);
	  break; 	     
	case '?':
	  /* getopt_long already printed an error message. */
	  break;
     
	default:
	  abort ();
	}
    }

  /* Print any remaining command line arguments (not options). */
  if (optind < argc)
    {
      printf ("non-option ARGV-elements: ");
      while (optind < argc)
	printf ("%s ", argv[optind++]);
      putchar ('\n');
    }


  double n1d=2*N1;
  double n2d=2*N2;
  /*find n_e*/
  double prop[4],n_es=0;
  double n_e=calculate_ne(N1,N2,t);   

  /*Set up Spikes*/ 
  double * spikes_vec = (double*) calloc (nspikes+1, sizeof(double));
  double * prob_vec = (double*) calloc (nspikes+1, sizeof(double));

  for(i = 1; i <= nspikes; i++){
    printf("Give s for spike %d:\n", i);
    scanf("%lf",&spikes_vec[i]);
  }
  spikes_vec[0]=0;

  double sum_prob=0;
  for(i = 1; i <= nspikes-1; i++){
    printf("Give probability of Spike %d:\n", i);
    scanf("%lf",&prob_vec[i]);
    sum_prob+=prob_vec[i];
  }
  prob_vec[nspikes]=1-sum_prob;
  prob_vec[0]=0;

  double **FVSX= calloc(maxnd+1, sizeof(double *));

  for(i = 1; i <= nspikes; i++){
    FVSX[i] = calloc(maxnd+1,  sizeof(double));
  }
  double * FV0 = (double*) calloc (maxnd+1, sizeof(double));

  /*Set up Tables*/

  get_data_path(data_path);
  set_up_file_name(N1, s_evaluated_vec_file_const, s_evaluated_vec_file);
  set_up_file_name(N1, s_range_file_const, s_range_file);
  get_s_evaluated_vec(s_evaluated_vec, &n_s_evaluated, &n_s_evaluated_file, 
		      s_evaluated_vec_file);
  get_s_ranges();

  if (!conpop)
    {
      set_up_file_name(N1, n2_evaluated_vec_file_const, n2_evaluated_vec_file);
      set_up_file_name(N1, t2_evaluated_vec_file_const, t2_evaluated_vec_file);
      set_up_file_name(N1, phase_1_dir_const, phase_1_dir);
      set_up_file_name(N1, phase_2_dir_const, phase_2_dir);

      get_int_evaluated_vec(t2_evaluated_vec,&n_t2_evaluated, &t2_lower,
			    &t2_step,&t2_evaluated_vec_file);
      get_int_evaluated_vec(n2_evaluated_vec,&n_n2_evaluated, &n2_lower,
			    &n2_step,&n2_evaluated_vec_file);
    }
  else
    {
      set_up_file_name(N1, "", const_pop_dir);
    }
 
  /* calculate neutral frequency vector*/ 
  load_FV(N1,N2,t,0.0,f0,FV0);

  /*Sampling of sites*/
  int * discrete0 = (int*) calloc (nalleles+2, sizeof(int));
  int * discrete1 = (int*) calloc (nalleles+2, sizeof(int));

  const gsl_rng_type * T;
  double selcoeff=0,uniform=0;

  gsl_rng_env_setup();
  T = gsl_rng_taus;
  gsl_rng  *rgen =gsl_rng_alloc(T);
  
  printf("%f,%f\n",gamma_alpha,gamma_beta);
  i=1;
  while (i<=sampleN)
    {// loop for sampling starts here
      i++;
      uniform=gsl_rng_uniform (rgen);
      double * FVS = (double*) calloc (maxnd+1, sizeof(double));
      selcoeff=0;
      switch(selmode)
	{
	case 0:case 1:
	  break;
	case 2:
	  selcoeff=gsl_ran_gamma (rgen, gamma_beta, 1/gamma_alpha);
	  break;
	case 3:
	  selcoeff=gsl_ran_beta (rgen, beta_alpha, beta_beta);
	  break;
	case 4:
	  if (uniform<=0.2) {selcoeff=gsl_ran_exponential(rgen,exp_mean);}
	  if (uniform>0.2) {selcoeff=gsl_ran_beta (rgen, beta_alpha, beta_beta);}
	  break;
	case 5:
	  if (uniform<=0.5) {selcoeff=gsl_ran_gamma (rgen, gamma_beta, 1/gamma_alpha);}
	  if (uniform>0.5) {selcoeff=gsl_ran_gamma (rgen, gamma2_beta, 1/gamma2_alpha);}
	  break;
	case 6:
	  if (uniform<=0.5) {selcoeff=gsl_ran_exponential(rgen, exp_mean);}
	  if (uniform>0.5) {selcoeff=gsl_ran_exponential (rgen, exp_mean2);}
	  break;
	case 7:
	  if (uniform<=0.5) {selcoeff=gsl_ran_gamma (rgen, gamma_beta, 1/gamma_alpha);}
	  if (uniform>0.5) {selcoeff=gsl_ran_exponential (rgen, exp_mean);}
	  break; 
	case 8:
	  if (uniform<=0.8) {selcoeff=gsl_ran_exponential(rgen,exp_mean);}
	  if (uniform>0.8) {selcoeff=gsl_ran_beta (rgen, beta_alpha, beta_beta);}
	  break;
	case 9:
          selcoeff=gsl_ran_exponential(rgen,exp_mean);
	  break;
	case 10:
          if (uniform<=0.5) {selcoeff=0.0;}
          if (uniform>0.5) {selcoeff=0.05;}
	  break;
	case 11:
          if (uniform<=(0.3)) {selcoeff=0.0;}
          if (uniform>0.33&&uniform<=0.67) {selcoeff=0.05;}
          if (uniform>0.67) {selcoeff=0.5;}
	  break;
    case 12:
          if (uniform<=0.3) {selcoeff=0;}//gsl_ran_gaussian (rgen,1e-5);}
          if (uniform>0.3&&uniform<=0.4) {selcoeff=-0.05;}//gsl_ran_gaussian (rgen,1e-5);selcoeff+=0.005;}
          if (uniform>0.4&&uniform<=0.6) {selcoeff=-0.5;}//gsl_ran_gaussian (rgen,1e-5);selcoeff+=0.05;}
          if (uniform>0.6) {selcoeff=-5;} //gsl_ran_gaussian (rgen,1e-5);selcoeff+=0.5;}
	  break;
	case 13:
          selcoeff=uniform;
	  break;
    case 14:
	  if (uniform<=0.5) {selcoeff=gsl_ran_gamma (rgen, gamma_beta, 1/gamma_alpha);}
	  if (uniform>0.5) {selcoeff=gsl_ran_beta (rgen, beta_alpha, beta_beta);}
	  break;
    case 15:
	  if (uniform<=0.2) {selcoeff=gsl_ran_gamma (rgen, gamma_beta, 1/gamma_alpha);}
	  if (uniform>0.2) {selcoeff=gsl_ran_beta (rgen, beta_alpha, beta_beta);}
	  break;
    case 16:
	  if (uniform<=0.8) {selcoeff=gsl_ran_gamma (rgen, gamma_beta, 1/gamma_alpha);}
	  if (uniform>0.8) {selcoeff=gsl_ran_beta (rgen, beta_alpha, beta_beta);}
	  break;
 	case 17:
          if (uniform<=(0.3)) {selcoeff=0.0;}
          if (uniform>0.33&&uniform<=0.67) {selcoeff=0.03;}
          if (uniform>0.67) {selcoeff=0.06;}
	  break;	
	case 18:
          if (uniform<=0.2) {selcoeff=sel_coeff1;}
          if (uniform>0.2&&uniform<=0.8) {selcoeff=sel_coeff2;}
          if (uniform>0.8) {selcoeff=sel_coeff3;}
	  break;
	case 19:
          if (uniform<=0.3) {selcoeff=0;}
          if (uniform>0.3) {selcoeff=0.05;}
	  break;
	case 20:
          if (uniform<=0.2) {selcoeff=0;}
          if (uniform>0.2&&uniform<=0.4) {selcoeff=0.05;}
          if (uniform>0.4) {selcoeff=0.5;}
	  break;
	case 21:
          if (uniform<=0.2) {selcoeff=0;}
          if (uniform>0.2&&uniform<=0.4) {selcoeff=0.02;}
          if (uniform>0.4&&uniform<=0.6) {selcoeff=0.05;}
          if (uniform>0.6) {selcoeff=0.1;}
          break;
    case 22:
	  selcoeff=exp_mean;
	  break;
    case 23:
	  selcoeff=gsl_ran_lognormal (rgen, gamma_alpha, gamma_beta);
	  break;
	case 24:
          if (uniform<=0.2) {selcoeff=sel_coeff1;}
          if (uniform>0.2) {selcoeff=sel_coeff2;}
	  break;	
    case 25:
          if (uniform<=0.5) {selcoeff=sel_coeff1;}
          if (uniform>0.5) {selcoeff=sel_coeff2;}
	  break;	
	case 26:
          if (uniform<=prob1) {selcoeff=sel_coeff1;}
          if (uniform>prob1) {selcoeff=sel_coeff2;}
	  break;
	case 27:
          if (uniform<=prob1) {selcoeff=sel_coeff1;}
          if (uniform>=prob1) {selcoeff=sel_coeff1;}
          if (uniform>prob1) {selcoeff=sel_coeff2;}
	  break;
	}//end switch

      //monitorinput(); 
      if(selcoeff>0){selcoeff=-selcoeff;}
      if (selcoeff<-100) {selcoeff=-100;}//free(FVS);continue;}//free(FVS);continue;
      //printf("%f\n",selcoeff);
      MEAN_S+=selcoeff;
      MEAN_S2+=-pow(selcoeff,2);
      MEAN_H+=1/selcoeff;
      

      if (selcoeff== 0)
        {
	  fix_prob += 0.5/n_e;
	}
      else
	{
	  fix_prob += kimura_fixation_prob(selcoeff, n_e);
	}       
      
      
      /*calculate relative proportions*/
      n_es=-n_e*selcoeff;
      if (n_es<=0.1){prop[0]++;}
      if (n_es<=1.0){prop[1]++;}
      if (n_es<=10.0){prop[2]++;}
      if (n_es<100.0){prop[3]++;}
      
      load_FV(N1,N2,t,selcoeff,f0,FVS);
     
      egf_scaling(N2,f0,FV0,FVS);
      gsl_ran_discrete_t *r= gsl_ran_discrete_preproc (n2d,FVS);
      double s1=gsl_ran_discrete (rgen, r);
      double prob=(double)(s1)/n2d;

      int success=gsl_ran_binomial(rgen,prob,nalleles);
      if (success==nalleles){discrete1[0]++;}else{
	discrete1[success]++;
      }
      free(FVS);
      gsl_ran_discrete_free(r);
      
    }//end sampling

  MEAN_S/=sampleN;
  MEAN_S2/=sampleN;
  MEAN_H/=sampleN;
  MEAN_H=1/MEAN_H;//Reciprocal of the mean of the reciprocals
  fix_prob/=sampleN;

  fix_prob *=2*n_e;

  prop[3]=(prop[3]-prop[2])/sampleN;
  prop[2]=(prop[2]-prop[1])/sampleN;
  prop[1]=(prop[1]-prop[0])/sampleN;
  prop[0]=prop[0]/sampleN;

  egf_scaling(N2,f0,FV0,FV0);
  binomial_sampling(N2,nalleles,sampleS,FV0,discrete0);//neutral

  /*Print Output*/

  printf("%f\t%f\t%f\n",MEAN_S,MEAN_S2,MEAN_H);
  //output_sfs_to_file_thanasis_format(nalleles,discrete0,discrete1,sfs_filename);
  output_sfs_to_file_peter_format2(nalleles,sampleS,sampleN,discrete1,discrete0,sfs_filename);

  FILE *file1= fopen(strcat(sfs_filename,".mean"), "w" );
  fprintf(file1,"%0.16E %0.16E %0.16E ",MEAN_S,MEAN_S2,MEAN_H);
  
  for(i = 0; i <= 3; i++){
    fprintf(file1,"%0.16E ",prop[i]);
  }
  fprintf(file1,"%0.16E ",fix_prob);
  fclose(file1);
  ////////////////////////////////////////////////////////////////////////////////
  
  free(FV0);
  gsl_rng_free (rgen);

  free(discrete0);
  free(discrete1);

  for(i = 1; i <= nspikes; i++){
    free(FVSX[i]);
  }
  free(FVSX);

  return 0;
}
