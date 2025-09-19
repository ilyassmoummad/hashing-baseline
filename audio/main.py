from retrieval import retrieval_knn_eval, retrieval_knn_eval_nopca
from get_model import get_audio_encoder, get_buckethead
from pca import AudioEncoderWithPCA, compute_pca
from get_data import get_dataloaders
from args import args
import numpy as np
import os


def main(args):
    # Load data and initialize encoder
    encoder, component, dim = get_audio_encoder(args)
    train_loader, test_loader = get_dataloaders(args)

    if not args.nopca:
        # Compute PCA
        _, _, pca_matrix, train_mean = compute_pca(encoder, component, train_loader, args)
        encoder_with_pca = AudioEncoderWithPCA(encoder, component, dim, pca_matrix, train_mean, args)

        # Run evaluation args.n_runs times with random bucketheads
        n_runs = args.n_runs
        all_results = {
            'map_cont': [],
            'map_asymhamming': [],
            'acc': [],
            'asymhamming_acc': []
        }

        for i in range(n_runs):
            if n_runs > 1:
                print(f"Run {i+1}/{n_runs}")
            buckethead = get_buckethead(args.bits).to(args.device)  # new random buckethead each run
            results = retrieval_knn_eval(encoder_with_pca, buckethead, component, train_loader, test_loader, args)

            # Store results
            for k in all_results:
                all_results[k].append(results[k])

        # Compute mean and std for each metric
        summary = {}
        for k, vals in all_results.items():
            vals = np.array(vals)
            summary[k] = (vals.mean(), vals.std())

        # Print mean ± std results
        print(f"Results over {n_runs} runs (mean ± std):")
        for k, (mean_val, std_val) in summary.items():
            print(f"{k}: {mean_val:.4f} ± {std_val:.4f}")

        if args.save:
            # Save to text file
            filename = f"results_{args.model}_bits{args.bits}_{args.dataset}.txt"
            save_path = os.path.join(args.results_dir, filename)
            with open(save_path, "w") as f:
                f.write(f"Results over {n_runs} runs (mean ± std):\n")
                for k, (mean_val, std_val) in summary.items():
                    f.write(f"{k}: {mean_val:.4f} ± {std_val:.4f}\n")

            print(f"Results saved to {save_path}")

    else:
        # Evaluate with original features once (no PCA)
        results = retrieval_knn_eval_nopca(encoder, component, train_loader, test_loader, args)
        print(results)

        if args.save:
            # Save single-run results to text file
            filename = f"results_{args.model}_original_{args.dataset}.txt"
            save_path = os.path.join(args.results_dir, filename)
            with open(save_path, "w") as f:
                f.write("Results (single run, original features):\n")
                for k, v in results.items():
                    f.write(f"{k}: {v:.4f}\n")

            print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main(args)
