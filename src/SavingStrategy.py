from typing import Optional, Union
import numpy as np
import flwr as fl
from flwr.common import Parameters, Scalar, FitRes
from flwr.server.strategy import FedYogi, FedAvg, FedAdam, FedTrimmedAvg, FedProx

class SaveModelYogi(FedYogi):
    """
    Custom Federated Learning strategy based on FedYogi.
    Extends FedYogi to save aggregated model weights after each round.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Aggregates model updates from clients and saves the aggregated weights.

        Args:
            server_round (int): The current round of federated learning.
            results: List of tuples containing client proxies and their fit results.
            failures: List of failures encountered during the round.

        Returns:
            tuple: Aggregated parameters and metrics.
        """

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated_ndarrays to disk
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"assets/models/latest_weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics
    

class SaveModelFedAvg(FedAvg):
    """
    Custom Federated Learning strategy based on FedAvg.
    Extends FedAvg to save aggregated model weights after each round.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Aggregates model updates from clients and saves the aggregated weights.

        Args:
            server_round (int): The current round of federated learning.
            results: List of tuples containing client proxies and their fit results.
            failures: List of failures encountered during the round.

        Returns:
            tuple: Aggregated parameters and metrics.
        """
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated_ndarrays to disk
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"assets/models/latest_weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics
    

class SaveModelAdam(FedAdam):
    """
    Custom Federated Learning strategy based on FedAdam.
    Extends FedAdam to save aggregated model weights after each round.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Aggregates model updates from clients and saves the aggregated weights.

        Args:
            server_round (int): The current round of federated learning.
            results: List of tuples containing client proxies and their fit results.
            failures: List of failures encountered during the round.

        Returns:
            tuple: Aggregated parameters and metrics.
        """
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated_ndarrays to disk
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"assets/models/latest_weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics
    

class SaveModelTrimmedAvg(FedTrimmedAvg):
    """
    Custom Federated Learning strategy based on FedTrimmedAvg.
    Extends FedTrimmedAvg to save aggregated model weights after each round.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Aggregates model updates from clients and saves the aggregated weights.

        Args:
            server_round (int): The current round of federated learning.
            results: List of tuples containing client proxies and their fit results.
            failures: List of failures encountered during the round.

        Returns:
            tuple: Aggregated parameters and metrics.
        """
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated_ndarrays to disk
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"assets/models/latest_weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics
    

class SaveModelProx(FedProx):
    """
    Custom Federated Learning strategy based on FedProx.
    Extends FedProx to save aggregated model weights after each round.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Aggregates model updates from clients and saves the aggregated weights.

        Args:
            server_round (int): The current round of federated learning.
            results: List of tuples containing client proxies and their fit results.
            failures: List of failures encountered during the round.

        Returns:
            tuple: Aggregated parameters and metrics.
        """
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated_ndarrays to disk
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"assets/models/latest_weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics



