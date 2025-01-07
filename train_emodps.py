import csv
import logging
import numpy as np
import os
import pandas as pd
import random

from platypus.core import Problem
from platypus.algorithms import EpsNSGAII
from platypus.evaluator import ProcessPoolEvaluator
from nile_rbf import TrainNile
from susquehanna_rbf import TrainSusquehanna

from rbf import rbf_functions
import csv
import argparse
from morl_baselines.common.performance_indicators import hypervolume, sparsity, cardinality
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--water-sim", type=str, help="Name of the reservoir", required=True)
    parser.add_argument("--num-of-obj", type=int, help="Number of Objectives", required=True)
    parser.add_argument("--nfes", type=int, help="Number of NFEs", required=True)
    parser.add_argument("--epsilons", type=float, nargs="+", help="Epsilons for epsilon-NSGA2", required=True)
    parser.add_argument("--seed", type=int, help="Random seed to use", default=42)
    parser.add_argument("--directory", type=str,help="Output directory", default='output')
    parser.add_argument("--entity", type=str,help="WANDB entity", required=False)
    parser.add_argument(
    "--ref-point", type=float, nargs="+", help="Reference point to use for the hypervolume calculation", required=True)
    parser.add_argument(
    "--log", type=bool, help="Log to wandb", default=False)

    return parser.parse_args()


class TrackProgress:
    def __init__(self, n_obj, env, epsilons, nfes, seed, ref_point, log, entity):
        self.nfe = []
        self.improvements = []
        self.objectives = {}
        self.ref_point = ref_point
        self.log=log
        if self.log:
            wandb.init(project="MORL4Water", entity=entity, name=f'{env}_{n_obj}_{nfes}_EMODPS_{epsilons}_{seed}', config={
                "algorithm": "EMODPS-EpsNSGAII",
                "num_of_obj": n_obj,
                "epsilons": epsilons  # or however many objectives you're using
            })

    def __call__(self, algorithm):
        self.nfe.append(algorithm.nfe)
        self.improvements.append(algorithm.archive.improvements)
        temp = {}
        objectives = []
        for i, solution in enumerate(algorithm.archive):
            temp[i] = np.array(solution.objectives)
            objectives.append(solution.objectives)
        if algorithm.nfe%1000==0:
            hv = hypervolume(ref_point=np.array(self.ref_point), points=objectives)
            sp = sparsity(objectives)
            cd = cardinality(objectives)
            print(f'NFE: {algorithm.nfe},Hypervolume: {hv}, sparsity: {sp}, cardinality: {cd}')

            if self.log:

                # Log metrics to wandb
                wandb.log({
                    
                    "Hypervolume": hv,
                    "Sparsity": sp,
                    "Cardinality": cd
                }, step=algorithm.nfe)
        
        self.objectives[algorithm.nfe] = pd.DataFrame.from_dict(temp, orient="index")

    def to_dataframe(self):
        df_imp = pd.DataFrame.from_dict(
            dict(nfe=self.nfe, improvements=self.improvements)
        )
        df_hv = pd.concat(self.objectives, axis=0)
        return df_imp, df_hv


def store_results(algorithm, track_progress, output_dir, rbf_name, seed_id):
    path_name = output_dir
    if not os.path.exists(path_name):
        try:
            os.mkdir(path_name)
        except OSError:
            print("Creation of the directory failed")

    with open(
        f"{output_dir}/{seed_id}_solution.csv",
        "w",
        encoding="UTF8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        for solution in algorithm.result:
            writer.writerow(solution.objectives)

    with open(
        f"{output_dir}/{seed_id}_variables.csv",
        "w",
        encoding="UTF8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        for solution in algorithm.result:
            writer.writerow(solution.variables)

    # save progress info
    df_conv, df_hv = track_progress.to_dataframe()
    df_conv.to_csv(f"{output_dir}/{seed_id}_convergence.csv")
    df_hv.to_csv(f"{output_dir}/{seed_id}_hypervolume.csv")


def main():
    args = parse_args()
    print(args)


    random.seed(args.seed)

    # RBF parameters - function to predict releases
    if args.water_sim=='nile':
        n_inputs = 5  # (storage in 4 reservoirs, month)
        n_outputs = 4
        n_rbfs = 9
        rbf = rbf_functions.RBF(n_rbfs, n_inputs, n_outputs, rbf_function=rbf_functions.original_rbf)
        reservoir = TrainNile(rbf, 240)
    elif args.water_sim=='susquehanna':
        n_inputs = 2  # (time, storage of Conowingo)
        n_outputs = 4
        n_rbfs = 6
        rbf = rbf_functions.RBF(n_rbfs, n_inputs, n_outputs, rbf_function=rbf_functions.original_rbf)
        reservoir = TrainSusquehanna(rbf, 2190)



    # Lower and Upper Bound for problem.types
    epsilons = list(args.epsilons)
    print(epsilons)
    n_decision_vars = len(rbf.platypus_types)

    n_objectives = args.num_of_obj


    problem = Problem(n_decision_vars, n_objectives)
    problem.types[:] = rbf.platypus_types
    problem.function = reservoir.run_episode

    for objective in range(n_objectives):
        problem.directions[objective] = Problem.MAXIMIZE 

    track_progress = TrackProgress(n_obj=n_objectives, env=args.water_sim, epsilons=epsilons, nfes=args.nfes, seed=args.seed, ref_point=args.ref_point, log=args.log, entity = args.entity)


    with ProcessPoolEvaluator() as evaluator:
        algorithm = EpsNSGAII(problem, epsilons=epsilons, evaluator=evaluator)
        algorithm.run(args.nfes, track_progress)

    store_results(
        algorithm, track_progress, args.directory, "rbf_original", args.seed
    )


if __name__ == "__main__": 
    logging.basicConfig(level=logging.INFO)
    main()
    wandb.finish()
