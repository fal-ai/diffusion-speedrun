import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from itertools import combinations
from datetime import datetime
import time


class GPUTester:
    def __init__(self):
        # Set environment variables for NCCL
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_P2P_DISABLE"] = "0"
        os.environ["NCCL_IB_HCA"] = "^mlx5"
        os.environ["NCCL_NET_GDR_LEVEL"] = "2"

        # Initialize CUDA
        # torch.cuda.init()
        self.n_gpus = torch.cuda.device_count()
        self.failed_pairs = []
        self.problematic_gpus = []

    def test_single_gpu(self, gpu_id):
        """Test if a single GPU is accessible"""
        try:
            with torch.cuda.device(gpu_id):
                # Try tensor allocation and operation
                x = torch.randn(1000, device=f"cuda:{gpu_id}")
                y = x * 2
                torch.cuda.synchronize(gpu_id)
                del x, y
                torch.cuda.empty_cache()
                return True
        except Exception as e:
            self.problematic_gpus.append(gpu_id)
            return False

    @staticmethod
    def test_gpu_pair(rank, world_size, gpu_pair, port):
        """Test NCCL communication between a pair of GPUs"""
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_pair[0]},{gpu_pair[1]}"

        try:
            # Initialize process group
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)

            # Create and move tensor to current GPU
            tensor = torch.randn(10000000, device=f"cuda:{rank}")

            # Perform all-reduce operation
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

            # Cleanup
            del tensor
            torch.cuda.empty_cache()
            dist.destroy_process_group()
            return True

        except Exception as e:
            if dist.is_initialized():
                dist.destroy_process_group()
            torch.cuda.empty_cache()
            print(f"Error testing GPUs {gpu_pair[0]} and {gpu_pair[1]}: {str(e)}")
            return False

    def run_tests(self):
        """Run all GPU tests"""
        print(f"\n{'='*60}")
        print(
            f"GPU Communication Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"{'='*60}")

        # Check if we have GPUs
        if self.n_gpus == 0:
            print("No CUDA devices found!")
            return

        print(f"\nFound {self.n_gpus} GPUs:")
        for i in range(self.n_gpus):
            try:
                name = torch.cuda.get_device_name(i)
                print(f"GPU {i}: {name}")
            except Exception as e:
                print(f"GPU {i}: Error getting device name")

        # Test individual GPUs
        print("\nTesting individual GPU accessibility:")
        for i in range(self.n_gpus):
            success = self.test_single_gpu(i)
            status = "✓" if success else "✗"
            print(f"{status} GPU {i}")

        if self.problematic_gpus:
            print(
                f"\nWarning: GPUs {self.problematic_gpus} showed individual access issues"
            )
            return

        # Test GPU pairs
        print("\nTesting NCCL communication between GPU pairs:")
        base_port = 12355
        gpu_pairs = list(combinations(range(self.n_gpus), 2))

        for idx, pair in enumerate(gpu_pairs):
            try:
                # Use different ports for each pair to avoid conflicts
                port = base_port + idx
                mp.spawn(self.test_gpu_pair, args=(2, pair, port), nprocs=2, join=True)
                print(f"✓ GPUs {pair[0]} and {pair[1]} - Success")
            except Exception as e:
                print(f"✗ GPUs {pair[0]} and {pair[1]} - Failed")
                self.failed_pairs.append(pair)

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print test summary"""
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")

        if not self.failed_pairs:
            print("✓ All GPU pairs can communicate successfully!")
            return

        print("\nProblematic GPU pairs:")
        for pair in self.failed_pairs:
            print(f"- GPUs {pair[0]} and {pair[1]}")

        # Analyze problematic GPUs
        problem_count = {}
        for pair in self.failed_pairs:
            for gpu in pair:
                problem_count[gpu] = problem_count.get(gpu, 0) + 1

        if problem_count:
            worst_gpu = max(problem_count.items(), key=lambda x: x[1])
            print(
                f"\nMost problematic GPU: {worst_gpu[0]} (failed in {worst_gpu[1]} pairs)"
            )


def main():
    mp.set_start_method("spawn", force=True)
    tester = GPUTester()
    tester.run_tests()


if __name__ == "__main__":
    main()
