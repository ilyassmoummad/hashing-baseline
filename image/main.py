from retrieval import retrieval_eval, retrieval_eval_nopca
from get_model import get_image_encoder, get_buckethead
from pca import ImageEncoderWithPCA, compute_pca
from get_data import get_dataloaders
from args import args
import numpy as np
import os


def main(args):
    # Load data and initialize encoder
    train_loader, database_loader, query_loader = get_dataloaders(args)
    encoder, dim = get_image_encoder(args)

    if not args.nopca:
        # Compute PCA
        _, _, pca_matrix, train_mean = compute_pca(encoder, train_loader, args)
        encoder_with_pca = ImageEncoderWithPCA(encoder, dim, pca_matrix, train_mean, args)

        # Run evaluation args.n_runs times with random bucketheads
        n_runs = args.n_runs
        all_results = {
            'map_cont': [],
            'map_asymhamming': [],
        }

        for i in range(n_runs):
            if n_runs > 1:
                print(f"Run {i+1}/{n_runs}")
            buckethead = get_buckethead(args.bits).to(args.device)  # new random buckethead each run
            results = retrieval_eval(encoder_with_pca, buckethead, database_loader, query_loader, args)

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
            pretrained_model_name = args.pretrained_model.lower()
            model_variants = {
                "simdinov2": {
                    "vitb16": "_vitb16-IN1k"
                },
                "dinov2": {
                    "vitb14": "_vitb14-LVD142",
                    "vitb16": "_vitb16-IN1k"
                }
            }
            model_name = None
            for base_model, variants in model_variants.items():
                if base_model in pretrained_model_name:
                    model_name = base_model
                    # Check for variant
                    for variant, suffix in variants.items():
                        if variant in pretrained_model_name:
                            model_name += suffix
                            break
                    break
            if model_name is None:
                raise ValueError(f"Unknown model in: {args.pretrained_model}")
            
            filename = f"results_{model_name}_bits{args.bits}_{args.dataset}.txt"
            save_path = os.path.join(args.results_dir, filename)
            os.makedirs(args.results_dir, exist_ok=True)

            with open(save_path, "w") as f:
                f.write(f"Results over {n_runs} runs (mean ± std):\n")
                for k, (mean_val, std_val) in summary.items():
                    f.write(f"{k}: {mean_val:.4f} ± {std_val:.4f}\n")

            print(f"Results saved to {os.path.join(args.results_dir, filename)}")

    else:
        # Evaluate with original features once (no PCA)
        results = retrieval_eval_nopca(encoder, database_loader, query_loader, args)
        print(results)

        if args.save:
            # Save single-run results to text file
            pretrained_model_name = args.pretrained_model.lower()
            model_variants = {
                "simdinov2": {
                    "vitb16": "_vitb16-IN1k"
                },
                "dinov2": {
                    "vitb14": "_vitb14-LVD142",
                    "vitb16": "_vitb16-IN1k"
                }
            }
            model_name = None
            for base_model, variants in model_variants.items():
                if base_model in pretrained_model_name:
                    model_name = base_model
                    # Check for variant
                    for variant, suffix in variants.items():
                        if variant in pretrained_model_name:
                            model_name += suffix
                            break
                    break
            if model_name is None:
                raise ValueError(f"Unknown model in: {args.pretrained_model}")
            
            filename = f"results_{model_name}_original_{args.dataset}.txt"
            save_path = os.path.join(args.results_dir, filename)
            with open(save_path, "w") as f:
                f.write("Results (single run, original features):\n")
                for k, v in results.items():
                    f.write(f"{k}: {v:.4f}\n")

            print(f"Results saved to {os.path.join(args.results_dir, filename)}")


if __name__ == "__main__":
    main(args)
