/*******************************************************************
 *                                                                 *
 *      FILE:      bco_greedy1.c                                   *
 *                                                                 *
 *      FUNCTION:  performs scheduling of independent tasks using  *
 *                 BCO general optimization metaheuristic          *
 *                 with matrix representation of schedule          *
 *                 task selection is random, machines - greedy     *
 *                 but minimizing the distance from y_gmin         *
 *                                                                 *
 *      ARGUMENT:  none                                            *
 *                                                                 *
 *      RETURN:    none                                            *
 *                                                                 *
 *      AUTHOR: Tatjana Davidovic, Milica Selmic                   *
 *              Mathematical Institute                             *
 *              Beograd, 2011.                                     *
 *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/time.h>
#include"omp.h"

#define MAIN
// Taca (v3): Koristim export OMP_NUM_THREADS=2 (smesten u hostname.sh), 
// umesto #define NUM_THREADS 2

#include "bco_defs.h"

/* Used function declarations */

extern int normal(int, int, int), bco_init(void);

// BEGINNING OF PROGRAM CODE

int main(void) {
	int i, j, b; /* local counters */
	int ii, jj, bb;
	int iii, jjj, bbb, bbbb;

	int n_comp_all, /* variables determining the number */
	n_comp_last, /* of components to be added in each */
	iii_count; /* forward pass */

	int aux; /* auxiliary variable */

	int best_bee; /* index of bee that generated best (partial) solution */
	int n_proc; /* number of processors excluded from selection */
	int it_over; /* break of iteration indicator */
	int R; /* number of recruiters (loyal bees) */
	int pass_count; /* forward pass counter */

	int loy[MAX_BEES_NO]; /* loyality indicator array */

	int sum; /* total sum of task execution times */
	int av_sum; /* average scheduling time = (int)1/p*sum+1 */
	char f_name[200], pom_name[100], stat_name[200]; /* name of example data file */
	char broj[5];

	int stop_ind; /* stopping criteria indicator: 0-no. iter; 1-no. unimp. iter; 2-runtime */
	int num_iter;
	double run_time; /* allowed CPU runtime in seconds */

	int n_it, /* current iteration number */
	n_nonimp_it; /* current number of iterations without improvement */
	int terminate; /* stopping criteria indicator */

	int n_tasks[MAX_BEES_NO]; /* number of unselected task for each bee */
	int roulete[MAX_BEES_NO]; /* roulet size for task selection by each bee */
	int Iroulete[MAX_BEES_NO][MAX_TASK_NO]; /* indices of non-selected tasks */

	int first[MAX_BEES_NO]; /* first task selection within iteration indicator for each bee */
	int max_y, min_y; /* aux. vars. representing max and min current load of processors */
	int min_y_p;   /*min_y_g is private variable for OpenMP parallel program, used in some parts as help variable */

	int value, bingo; /* integer roulet variables */
	double dvalue, dbingo; /* probability roulet variables */
	double maxV_norm; /* maximum of normalized values for bees partial solutions */
	double sum_V_loy; /* sum of normalized values for bees partial solutions */

	double V_norm[MAX_BEES_NO]; /* normalized values for bees partial solutions */
	double p_loy[MAX_BEES_NO]; /* probability that bee is loyal */
	double Droulete[MAX_BEES_NO]; /* probability that recruiter is chosen */

	double V[MAX_BEES_NO][MAX_PROC_NO]; /* normalized values of processor loads for each bee */
	double sum_V[MAX_BEES_NO]; /* cumulative normalized values of processor loads for each bee */
	double p_proc[MAX_BEES_NO][MAX_PROC_NO]; /* probability of selection a processor for each bee */
	
	int ID, total_threads;
	//Taca (v3): Iskljuceno jer korsistim export OMP_NUM_THREADS=2 umesto : omp_set_num_threads(NUM_THREADS);
// Time variables.
	time_t timer;
	char *timeline;
	clock_t t1, t2;
	double t, min_t;
// OpenMP time variables
	double t_op;

	struct rusage r;
	time_t ctime;

	/* Beginning of program code */

	time(&ctime);
	srand(ctime); /* initialization of random number generator */
//      srand(1); /* initialization of random number generator */

	/* Reading input parameters */

	FILE *fin = fopen("input.dat", "r");

	//strcpy(f_name,"/home/tanjad/scheduling/indep/");
	strcpy(f_name, "");
	fscanf(fin, "%s %*s\n", pom_name);
	strcat(f_name, pom_name);
	strcat(f_name, ".dat");
	fscanf(fin, "%d %*s\n", &stop_ind);
	if (stop_ind < 2)
		fscanf(fin, "%d %*s \n", &num_iter);
	else
		fscanf(fin, "%lf %*s \n", &run_time);
	fscanf(fin, "%d %*s\n", &bees);
	fscanf(fin, "%d %*s\n", &NC);

	fclose(fin);

	strcpy(stat_name, pom_name);
	strcat(stat_name, "_bco_greedy1_B");
	sprintf(broj, "%d", bees);
	strcat(stat_name, broj);
	strcat(stat_name, "NC");
	sprintf(broj, "%d", NC);
	strcat(stat_name, broj);
	strcat(stat_name, "it");
	if (stop_ind == 2)
		num_iter = 500;
	sprintf(broj, "%d", num_iter);
	strcat(stat_name, broj);
	strcat(stat_name, ".dat");
//   printf("Ime fajla sa duzinama raspodele je %s\n",stat_name);

	/* Reading example parameters */

	FILE *name = fopen(f_name, "r");

	fscanf(name, "%*s %*s %*s\n %*s %*s %*s %*s %*s %*s\n");
	fscanf(name, "%*s %*s %*s %d;\n", &n);
	fscanf(name, "%*s %*s %*s %d;\n", &p);

	if ((L = (int *) malloc(n * sizeof(int))) == NULL)
		exit(MEM_ERROR);

	fscanf(name, "%*s %*s %*s %*s\n");
	for (i = 0; i < n; i++)
		fscanf(name, "%*d %d;\n", &L[i]);

	fclose(name);

	/* Memory allocation and initialization */

	if ((aux = bco_init()) == MEM_ERROR)
		exit(MEM_ERROR);

	sum = 0;
	for (i = 0; i < n; i++)
		sum += L[i];

	av_sum = (int) ((double) sum / (double) p) + 1;
	y_gmin = sum;
	n_comp_all = n / NC;
	if (NC * n_comp_all < n)
		n_comp_all += 1;
//   printf("n/NC = %d\n",n/NC);
//   printf("n_comp_all*NC = %d\n",n_comp_all*n/NC);
//   printf("n_comp_all = %d\n", n_comp_all);
	iii_count = n_comp_all;
	n_comp_last = n - n_comp_all * (NC - 1);
//   printf("n_comp_last = %d\n", n_comp_last);
//   printf("iii_count = %d\n", iii_count);

	/* Start of the scheduling process */

	n_it = n_nonimp_it = 0;
	terminate = 0;

	/* Time measurement beginning */

	t1 = clock();
#pragma omp parallel default(shared) private(b,bb,bbb,bbbb,i,j,ID,iii,aux,bingo,dbingo,value,ii,jj,min_y_p,t_op) //shared(roulete,bees,sum,n,n_tasks,first,Iroulete,y,o,s,p)
		{

	ID = omp_get_thread_num();
	t_op = omp_get_wtime ( );

	while (!terminate) { /* until the stopping criteria is satisfied */
#pragma omp barrier
		pass_count = 0;
		n_it++; /* new iteration starts */
//printf("Iteracija %d\n",n_it);
		// int ID;

#pragma omp for  
		for (b = 0; b < bees; b++) {
			//printf("Thread ID = %d b = %d\n",ID,b);
			roulete[b] = sum; /* initialization of data for selecting tasks */
			n_tasks[b] = n;
			first[b] = 1;
			for (i = 0; i < n; i++)
				Iroulete[b][i] = i;

			for (j = 0; j < p; j++)
				y[b][j] = 0;

			for (j = 0; j < p; j++)
				o[b][j] = 0;

			for (j = 0; j < MAX_PROC_NO; j++)
				for (i = 0; i < MAX_TASK_NO; i++)
					s[b][j][i] = -1;
			} /* for (b) */


		i = 0;
		it_over = 0;
		iii_count = n_comp_all;
#pragma omp barrier

		do /* until there are tasks to be scheduled */
		{

			if (i + n_comp_all <= n)
				i += n_comp_all;
			else {
				i += n_comp_last;
				iii_count = n_comp_last;
			}
			//  printf("i = %d     iii_count = %d\n", i, iii_count);

			pass_count++;

#pragma omp barrier
//printf("Forward pass %d\n",pass_count);


#pragma omp for 
			for (b = 0; b < bees; b++) /* FORWARD PASS */
			{
				for (iii = 0; iii < iii_count; iii++) /* NC tasks are scheduled per a single forward pass */
				{
					aux = rand(); /* randomly chose a task out of roulete weel */
					bingo = normal(aux, 1, roulete[b]);
					ii = 0;
					value = L[Iroulete[b][0]]; /* a task ii will be selected for scheduling */
					while (ii < n_tasks[b] && bingo > value)
						value += L[Iroulete[b][++ii]];

					if (first[b]) /* randomly choose processor using roulete */
					{
						aux = rand(); /* at the beginning all processors have */
						jj = normal(aux, 0, p - 1); /* equal probability to be */
						first[b] = 0; /* selected */
					} /* if (first[b]) */
					else /* select the processor whose occupation will be */
					{ /* closest to the current best solution */
//Taca (v1): #pragma omp critical - Not needed because of using of min_y_p which is used only in this for-cycle
					{ /* begin CRITICAL */
						min_y_p = sum;
						for (j = 0; j < p; j++)
							if (av_sum - (y[b][j] + L[Iroulete[b][ii]]) > 0 && //av_sum umesto y_gmin
								av_sum - (y[b][j] + L[Iroulete[b][ii]])	< min_y_p) 
								{
								min_y_p = av_sum - (y[b][j] +  L[Iroulete[b][ii]]); //av_sum umesto y_gmin
								jj = j;
								}
					} /* end CRITICAL */
					} /* else (first[b]) */
						
					/* task Iroulete[b][ii] is scheduled to proc. jj by bee b */
					s[b][jj][o[b][jj]++] = Iroulete[b][ii]; /* task ii is scheduled to proc. jj and all data is updated */
					y[b][jj] += L[Iroulete[b][ii]];
					roulete[b] -= L[Iroulete[b][ii]];
					Iroulete[b][ii] = Iroulete[b][--n_tasks[b]];
				} /* for (iii) */
				
			}/* for(b) */

			/* BACKWARD PASS */


#pragma omp for
			for (b = 0; b < bees; b++)
					y_max[b] = 0;
#pragma omp for
			for (b = 0; b < bees; b++) {

				/* evaluation of all partial solutions */

				max_y = 0;
				min_y = sum;
				for (j = 0; j < p; j++)
					if (y_max[b] < y[b][j])
						y_max[b] = y[b][j];
			} /* for (b) */


#pragma omp master
			{ /* begin #pragma omp master */
				
			for (b = 0; b < bees; b++) {
				if (max_y < y_max[b])
					max_y = y_max[b];
				if (min_y > y_max[b]) {
					best_bee = b;
					min_y = y_max[b];
				}
			} /* for (b) */
					
			} /* end #pragma omp master */
#pragma omp barrier


			if (i < n * NC) {


				if (max_y == min_y) {

#pragma omp for
				for (b = 0; b < bees; b++)
					loy[b] = -1;

			} else {
  
				dvalue = (double) (max_y - min_y);
				maxV_norm = (double) 0.0;
#pragma omp barrier


//Taca (v2): #pragma omp for - using "master" option here becasue I don't want critical point here
#pragma omp master
			{ /* begin #pragma omp master */
				for (b = 0; b < bees; b++) {
					loy[b] = 0;

					V_norm[b] = (double) (max_y - y_max[b]) / dvalue;
//Taca (v2): #pragma omp critical
					{ /* begin CRITICAL */
					if (maxV_norm < V_norm[b])
						maxV_norm = V_norm[b];
					} /* end CRITICAL*/

					}
			} /* end #pragma omp master */		

				sum_V_loy = (double) 0.0;
				R = 0;
#pragma omp barrier

#pragma omp for
				for (b = 0; b < bees; b++) {
					if (y_max[b] > y_gmin)
						p_loy[b] = (double) 0.0;
					else
						p_loy[b] = exp(
								-((double) (maxV_norm - V_norm[b])
										/ (double) pass_count));
					aux = rand();
					dbingo = (double) aux / (double) RAND_MAX;
					if (dbingo < p_loy[b]) {
#pragma omp critical
					{ /* begin CRITICAL */
					R++;
					sum_V_loy += V_norm[b];
					} /* end CRITICAL*/
					loy[b] = -1;

					}
				} /* for[b] */



						/* calculation of probability that loyal bee is chosen */

				if (R > 0) {

#pragma omp master
				{ /*begin #pragma omp master */

				bb= 0;


				for (b = 0; b < bees; b++)
					if (loy[b] == -1)
						{
						Droulete[bb++] = (double) (V_norm[b])
								/ (double) sum_V_loy;
						}
				} /* end #pragma omp master */
#pragma omp barrier

							
							
							/* selection of a recruiter by each follower */


				for (b = 0; b < bees; b++)

					if (!loy[b]) {

#pragma omp master
					{ /*begin #pragma omp master */


					aux = rand();/* for each non-loyal bee */
					dbingo = (double) aux / (double) RAND_MAX;

					bb = 0;
					dvalue = Droulete[0]; /* the most suitable recruiter is selected */


					while (bb < R && dbingo > dvalue)
						dvalue += Droulete[++bb];
						bbb = -1;
						bbbb = -1;

						while (bbbb < bb && bbb < bees)
							if (loy[++bbb] == -1)
							bbbb++;


					for (j = 0; j < p; j++) /* copying schedule from recruiter to follower */
						{
						for (ii = 0; ii < o[bbb][j]; ii++)
							s[b][j][ii] = s[bbb][j][ii];
							if (o[b][j] > o[bbb][j])
								for (ii = o[bbb][j]; ii < o[b][j]; ii++)
									s[b][j][ii] = -1;
									o[b][j] = o[bbb][j];
									y[b][j] = y[bbb][j];
						} /* for (j) */
					y_max[b] = y_max[bbb];
					n_tasks[b] = n_tasks[bbb];
					roulete[b] = roulete[bbb];
					for (iii = 0; iii < n; iii++)
						Iroulete[b][iii] = Iroulete[bbb][iii];
					} /* end #pragma omp master */

					} /* end if (!loy[b]) */


				} /* if (R > 0) */
				else
					it_over = 1;
				}
				//} /* else (y_max==y_min) */


			} /* if (i < (n-1)*NC */


		} while (i < n && !it_over); /* end of one iteration */
		//}//End of OMP_Parallel Section
#pragma omp barrier
		
		/* checking the quality of the best solution per iteration */
#pragma omp master
		{
		if (y_max[best_bee] < y_gmin) {
			y_gmin = y_max[best_bee];
			av_sum = y_gmin;
			n_nonimp_it = 0;
			for (j = 0; j < p; j++) {
				for (i = 0; i < o[best_bee][j]; i++)
					sgmin[j][i] = s[best_bee][j][i];
				ogmin[j] = o[best_bee][j];
				ygmin[j] = y[best_bee][j];
			}

//Taca (v3): Koristim Openm wtime() funkciju za merenje vremena, a ne -> 
			//t2 = clock();
			//min_t = (double) (t2 - t1) / CLOCKS_PER_SEC;
			min_t = omp_get_wtime () - t_op;

		} else
			n_nonimp_it++;

		/* checking if stopping criteria is satisfied */

		switch (stop_ind) {
		case 0:
			if (n_it >= num_iter)
				terminate = 1;
			break;
		case 1:
			if (n_nonimp_it >= num_iter)
				terminate = 1;
			break;
		case 2:
			t2 = clock();
			t = (double) (t2 - t1) / CLOCKS_PER_SEC;
			if (t > run_time)
				terminate = 1;
			break;

		} /* switch (stop_ind) */
		}/* #pragma omp master */

#pragma omp barrier
		//}/* omp parallel*/
	} /* while (!terminate) */
	total_threads = omp_get_num_threads();
	t_op = omp_get_wtime ( ) - t_op;
	

	printf ("\n: OpenMP time_th(%d) - %lf",ID, t_op);
	
	}//End of OMP_Parallel Section

	/* printing of the minimum obtained schedule */
	printf ("\nNumber of threads: %d \n", total_threads);
	printf ("Bees = %d \n", bees );
	printf ("NC = %d \n", NC );
	printf ("------------------------------------------------------------------ \n");
	
	printf("Pronadjena minimalna duzina raspodele je: %d\n", y_gmin);
	printf("matrica raspodele zadatka je:\n");

	for (j = 0; j < p; j++) {
		for (i = 0; i < ogmin[j]; i++)
			printf("%d  ", sgmin[j][i]);
		printf("\n");
	}

	t2 = clock();
	t = (double) (t2 - t1) / CLOCKS_PER_SEC;

	printf("BCO radio %lf sekundi i izvrsio %d itercija.\n", t, n_it);
	printf("Vreme do pronalazenja minimalne raspodele: %lf\n", min_t);

	printf ("------------------------------------------------------------------ \n");
	printf ("------------------------------------------------------------------ \n");
	
	FILE *stat = fopen(stat_name, "a");
	fprintf(stat, "%d    %lf\n", y_gmin, min_t);
	fclose(stat);

	exit(OK);
}

