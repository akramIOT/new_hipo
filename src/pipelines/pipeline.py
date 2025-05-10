"""
Pipeline module for ML workflows.
"""
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Union
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class Pipeline:
    """Pipeline for ML workflows."""

    def __init__(self, name: str):
        """Initialize pipeline.

        Args:
            name: Pipeline name.
        """
        self.name = name
        self.steps = []
        self.data = {}
        self.metadata = {"name": name, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": [], "metrics": {}}

    def add_step(self, name: str, function: Callable, **kwargs) -> "Pipeline":
        """Add a step to the pipeline.

        Args:
            name: Step name.
            function: Function to call for this step.
            **kwargs: Additional arguments to pass to the function.

        Returns:
            Self for method chaining.
        """
        self.steps.append({"name": name, "function": function, "kwargs": kwargs})
        self.metadata["steps"].append(name)
        return self

    def run(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the pipeline.

        Args:
            input_data: Optional input data to start the pipeline.

        Returns:
            Dictionary containing pipeline results.
        """
        if input_data:
            self.data = input_data.copy()

        logger.info(f"Running pipeline: {self.name}")
        start_time = time.time()

        for i, step in enumerate(self.steps):
            step_name = step["name"]
            function = step["function"]
            kwargs = step["kwargs"]

            logger.info(f"Running step {i+1}/{len(self.steps)}: {step_name}")
            step_start_time = time.time()

            try:
                # Merge data and kwargs
                params = {**self.data, **kwargs}
                # Run the function
                result = function(**params)
                # Update data with result
                if isinstance(result, dict):
                    self.data.update(result)
                else:
                    self.data[step_name] = result

                step_time = time.time() - step_start_time
                logger.info(f"Step {step_name} completed in {step_time:.2f} seconds")

            except Exception as e:
                logger.error(f"Error in step {step_name}: {e}")
                raise RuntimeError(f"Pipeline failed at step {step_name}") from e

        total_time = time.time() - start_time
        logger.info(f"Pipeline {self.name} completed in {total_time:.2f} seconds")

        # Record pipeline metrics
        self.metadata["metrics"]["total_time"] = total_time
        self.metadata["metrics"]["steps_count"] = len(self.steps)

        return self.data

    def get_data(self) -> Dict[str, Any]:
        """Get pipeline data.

        Returns:
            Dictionary containing pipeline data.
        """
        return self.data

    def get_metadata(self) -> Dict[str, Any]:
        """Get pipeline metadata.

        Returns:
            Dictionary containing pipeline metadata.
        """
        return self.metadata

    def save_results(self, output_dir: str, filename: Optional[str] = None) -> str:
        """Save pipeline results.

        Args:
            output_dir: Output directory.
            filename: Optional filename. If None, use pipeline name.

        Returns:
            Path to saved results.
        """
        if filename is None:
            filename = f"{self.name}_results.json"

        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare results (exclude non-serializable objects)
        results = {"metadata": self.metadata, "data": {}}

        for key, value in self.data.items():
            if isinstance(value, (str, int, float, bool, list, dict)):
                results["data"][key] = value
            elif isinstance(value, (pd.DataFrame, pd.Series)):
                # Save pandas objects to separate files
                data_filename = f"{self.name}_{key}.parquet"
                data_path = Path(output_dir) / data_filename

                if isinstance(value, pd.DataFrame):
                    value.to_parquet(data_path)
                elif isinstance(value, pd.Series):
                    value.to_frame().to_parquet(data_path)

                results["data"][key] = {"type": "pandas", "filename": data_filename}
            elif isinstance(value, np.ndarray):
                # Save numpy arrays to separate files
                array_filename = f"{self.name}_{key}.npy"
                array_path = Path(output_dir) / array_filename
                np.save(array_path, value)

                results["data"][key] = {"type": "numpy", "filename": array_filename, "shape": value.shape}
            else:
                # For non-serializable objects, just store type information
                results["data"][key] = {"type": str(type(value))}

        # Save the main results file
        import json

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        return str(output_path)
