import inspect
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, List, Set, Type, Any
from dataclasses import dataclass
import json
import warnings

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


# GRAPH AND SERIALISATION HELPERS
class PipelineRegistry:
    """Registry for pipeline classes that can be used by nodes"""
    _pipelines: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, pipeline_class: Any) -> None:
        cls._pipelines[name] = pipeline_class

    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        return cls._pipelines.get(name)

    @classmethod
    def clear(cls) -> None:
        cls._pipelines.clear()

    @classmethod
    def list_registered(cls) -> List[str]:
        return list(cls._pipelines.keys())


@dataclass
class Connection:
    from_node: str
    from_socket: str
    to_node: str
    to_socket: str

    def to_dict(self) -> dict:
        return {
            "from_node": self.from_node,
            "from_socket": self.from_socket,
            "to_node": self.to_node,
            "to_socket": self.to_socket
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Connection':
        return cls(**data)


# ABSTRACT NODE CLASS
class Node(ABC):
    def __init__(self, node_id: str, input_sockets: Set[str], output_sockets: Set[str]):
        self.id = node_id
        self._input_socket_names = input_sockets
        self._output_socket_names = output_sockets
        self.input_sockets: Dict[str, np.ndarray] = {name: None for name in input_sockets}
        self.output_sockets: Dict[str, np.ndarray] = {name: None for name in output_sockets}

    @abstractmethod
    def process(self, PipelineRegistery) -> None:
        pass

    def set_input(self, socket_name: str, value: np.ndarray) -> None:
        if socket_name not in self._input_socket_names:
            raise ValueError(f"Invalid input socket: {socket_name}")
        self.input_sockets[socket_name] = value

    def get_output(self, socket_name: str) -> Optional[np.ndarray]:
        if socket_name not in self._output_socket_names:
            raise ValueError(f"Invalid output socket: {socket_name}")
        return self.output_sockets.get(socket_name)

    @abstractmethod
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.__class__.__name__,
            "input_sockets": list(self._input_socket_names),
            "output_sockets": list(self._output_socket_names)
        }

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> 'Node':
        pass

    @classmethod
    @abstractmethod
    def get_required_pipelines(cls) -> Set[str]:
        """Return set of pipeline class names required by this node"""
        return set()


# PIPELINE NODES
class LoadEmbeddingNode(Node):
    def __init__(self, node_id: str, audio_path: str, pipeline_name: str):
        super().__init__(
            node_id,
            input_sockets=set(),
            output_sockets={"embedding"}
        )
        self.audio_path = audio_path
        self.pipeline_name = pipeline_name
        self._pipeline = None

    def process(self, PipelineRegistery) -> None:
        if self._pipeline is None:
            pipeline_class = PipelineRegistry.get(self.pipeline_name)
            if pipeline_class is None:
                raise RuntimeError(f"Pipeline '{self.pipeline_name}' not registered, please add the StyleTTS pipeline "
                                   f"to the registry")
            self._pipeline = pipeline_class

        # Use pipeline to load embedding
        self.output_sockets["embedding"] = self._pipeline.compute_style(self.audio_path).cpu().numpy()

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "audio_path": self.audio_path,
            "pipeline_name": self.pipeline_name
        })
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'LoadEmbeddingNode':
        return cls(data["id"], data["audio_path"], data["pipeline_name"])

    @classmethod
    def get_required_pipelines(cls) -> Set[str]:
        return {"StyleTTSPipeline"}


class SaveEmbeddingNode(Node):
    def __init__(self, node_id: str, output_path: str):
        super().__init__(
            node_id,
            input_sockets={"embedding"},
            output_sockets=set()
        )
        self.output_path = output_path

    def process(self, PipelineRegistery) -> None:
        # Ensure the embedding input is connected
        if self.input_sockets["embedding"] is None:
            raise ValueError("Input socket 'embedding' is not connected")

        # Retrieve the embedding
        embedding = self.input_sockets["embedding"]

        # Convert embedding to a torch tensor
        tensor_embedding = torch.from_numpy(embedding)

        # Ensure the output path has the .emb extension
        if not self.output_path.endswith(".emb"):
            self.output_path += ".emb"

        # Save the tensor to the specified file
        torch.save(tensor_embedding, self.output_path)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "output_path": self.output_path
        })
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'SaveEmbeddingNode':
        return cls(data["id"], data["output_path"])

    @classmethod
    def get_required_pipelines(cls) -> Set[str]:
        return set()


class InferenceNode(Node):
    def __init__(self, node_id: str, test_text: str, out_path: str, pipeline_name: str,
                 diffusion_steps: int = 16,
                 alpha: float = 0.3,
                 beta: float = 0.4,
                 embedding_scale: float = 1.2,
                 speed: float = 1):
        super().__init__(
            node_id,
            input_sockets={"embedding"},
            output_sockets={"audio_out"}
        )
        self.out_path = out_path
        self.pipeline_name = pipeline_name
        self._pipeline = None
        self.test_text = test_text
        self.diffusion_steps = diffusion_steps
        self.alpha = alpha
        self.beta = beta
        self.embedding_scale = embedding_scale
        self.speed = 1

    def process(self, PipelineRegistery) -> None:
        if self._pipeline is None:
            pipeline_class = PipelineRegistry.get(self.pipeline_name)
            if pipeline_class is None:
                raise RuntimeError(f"Pipeline '{self.pipeline_name}' not registered, please add the StyleTTS pipeline "
                                   f"to the registry")
            self._pipeline = pipeline_class

        if self.input_sockets["embedding"] is None:
            raise ValueError("Input socket (embedding) not connected")

        # Use pipeline to load embedding
        audOut = self._pipeline.generate(self.test_text,
                                         torch.from_numpy(self.input_sockets["embedding"]).to(device),
                                         diffusion_steps=self.diffusion_steps,
                                         alpha=self.alpha,
                                         beta=self.beta,
                                         embedding_scale=self.embedding_scale,
                                         output_file_path=self.out_path,
                                         speed=self.speed,
                                         language="en")
        self.output_sockets["audio_out"] = audOut

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "out_path": self.out_path,
            "pipeline_name": self.pipeline_name,
            "test_text": self.test_text,
            "diffusion_steps": self.diffusion_steps,
            "alpha": self.alpha,
            "beta": self.beta,
            "embedding_scale": self.embedding_scale,
            "speed": self.speed
        })
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'InferenceNode':
        return cls(data["id"],
                   data["test_text"],
                   data["out_path"],
                   data["pipeline_name"],
                   diffusion_steps=data["diffusion_steps"],
                   alpha=data["alpha"],
                   beta=data["beta"],
                   embedding_scale=data["embedding_scale"],
                   speed=data["speed"])

    @classmethod
    def get_required_pipelines(cls) -> Set[str]:
        return {"StyleTTSPipeline"}


# GENERAL NODES

class NormalizeNode(Node):
    def __init__(self, node_id: str):
        super().__init__(
            node_id,
            input_sockets={"embedding"},
            output_sockets={"output"}
        )

    def process(self, PipelineRegistery) -> None:
        if self.input_sockets["embedding"] is None:
            raise ValueError("Input socket not connected")

        embedding = self.input_sockets["embedding"]

        # explicit function to normalize array
        def normalize(arr, t_min, t_max):
            norm_arr = []
            diff = t_max - t_min
            diff_arr = max(arr) - min(arr)
            for i in arr:
                temp = (((i - min(arr)) * diff) / diff_arr) + t_min
                norm_arr.append(temp)
            return norm_arr

        normalized = normalize(embedding, 0, 1)

        self.output_sockets["output"] = np.asarray(normalized)

    def to_dict(self) -> dict:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> 'NormalizeNode':
        return cls(data["id"])

    @classmethod
    def get_required_pipelines(cls) -> Set[str]:
        return set()


class WeightedAverageNode(Node):
    def __init__(self, node_id: str, weight: float):
        super().__init__(
            node_id,
            input_sockets={"embedding1", "embedding2"},
            output_sockets={"output"}
        )

        self.weight = weight

    def process(self, PipelineRegistery) -> None:
        for inputSock in self.input_sockets.keys():
            if self.input_sockets[inputSock] is None:
                raise ValueError(f"Input socket ({inputSock}) not connected")

        embedding1 = self.input_sockets["embedding1"]
        embedding2 = self.input_sockets["embedding2"]

        weight = self.weight
        weight2 = 100 - weight

        weights = [weight, weight2]
        values = [embedding1, embedding2]
        weighted_average = sum(weight * value for weight, value in zip(weights, values)) / sum(weights)

        self.output_sockets["output"] = weighted_average

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "weight": self.weight
        })
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'WeightedAverageNode':
        return cls(data["id"], data["weight"])

    @classmethod
    def get_required_pipelines(cls) -> Set[str]:
        return set()


class AdditionNode(Node):
    def __init__(self, node_id: str):
        super().__init__(
            node_id,
            input_sockets={"embedding1", "embedding2"},
            output_sockets={"output"}
        )

    def process(self, PipelineRegistery) -> None:
        embedding1 = self.input_sockets["embedding1"]
        embedding2 = self.input_sockets["embedding2"]

        self.output_sockets["output"] = embedding1 + embedding2

    def to_dict(self) -> dict:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> 'AdditionNode':
        return cls(data["id"])

    @classmethod
    def get_required_pipelines(cls) -> Set[str]:
        return set()


class StyleTransferNode(Node):
    def __init__(self, node_id: str):
        super().__init__(
            node_id,
            input_sockets={"acoustic", "prosody"},
            output_sockets={"output"}
        )

    def process(self, PipelineRegistery) -> None:
        acoustic = self.input_sockets["acoustic"]
        prosody = self.input_sockets["prosody"]

        acoustic_size = acoustic.shape[1]  # Number of columns in ref_s
        acoustic_isolated = acoustic[:, :acoustic_size]

        prosody_size = prosody.shape[1]
        prosody_isolated = prosody[:, prosody_size:]

        self.output_sockets["output"] = np.concatenate([acoustic_isolated, prosody_isolated], axis=1)

    def to_dict(self) -> dict:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> 'StyleTransferNode':
        return cls(data["id"])

    @classmethod
    def get_required_pipelines(cls) -> Set[str]:
        return set()


class SubtractionNode(Node):
    def __init__(self, node_id: str):
        super().__init__(
            node_id,
            input_sockets={"embedding1", "embedding2"},
            output_sockets={"output"}
        )

    def process(self, PipelineRegistery) -> None:
        embedding1 = self.input_sockets["embedding1"]
        embedding2 = self.input_sockets["embedding2"]

        self.output_sockets["output"] = embedding1 - embedding2

    def to_dict(self) -> dict:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> 'SubtractionNode':
        return cls(data["id"])

    @classmethod
    def get_required_pipelines(cls) -> Set[str]:
        return set()


class MultiplicationNode(Node):
    def __init__(self, node_id: str):
        super().__init__(
            node_id,
            input_sockets={"embedding1", "embedding2"},
            output_sockets={"output"}
        )

    def process(self, PipelineRegistery) -> None:
        embedding1 = self.input_sockets["embedding1"]
        embedding2 = self.input_sockets["embedding2"]

        self.output_sockets["output"] = embedding1 * embedding2

    def to_dict(self) -> dict:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> 'MultiplicationNode':
        return cls(data["id"])

    @classmethod
    def get_required_pipelines(cls) -> Set[str]:
        return set()


class DivisionNode(Node):
    def __init__(self, node_id: str):
        super().__init__(
            node_id,
            input_sockets={"embedding1", "embedding2"},
            output_sockets={"output"}
        )

    def process(self, PipelineRegistery) -> None:
        embedding1 = self.input_sockets["embedding1"]
        embedding2 = self.input_sockets["embedding2"]

        if embedding2 == 0:
            raise ZeroDivisionError("Division by zero is not allowed.")

        self.output_sockets["output"] = embedding1 / embedding2

    def to_dict(self) -> dict:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> 'DivisionNode':
        return cls(data["id"])

    @classmethod
    def get_required_pipelines(cls) -> Set[str]:
        return set()


class MeanNode(Node):
    def __init__(self, node_id: str):
        super().__init__(
            node_id,
            input_sockets={"embedding1", "embedding2"},
            output_sockets={"output"}
        )

    def process(self, PipelineRegistery) -> None:
        embedding1 = self.input_sockets["embedding1"]
        embedding2 = self.input_sockets["embedding2"]

        self.output_sockets["output"] = (embedding1 + embedding2) / 2

    def to_dict(self) -> dict:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> 'MeanNode':
        return cls(data["id"])

    @classmethod
    def get_required_pipelines(cls) -> Set[str]:
        return set()


class MinNode(Node):
    def __init__(self, node_id: str):
        super().__init__(
            node_id,
            input_sockets={"embedding1", "embedding2"},
            output_sockets={"output"}
        )

    def process(self, PipelineRegistery) -> None:
        embedding1 = self.input_sockets["embedding1"]
        embedding2 = self.input_sockets["embedding2"]

        self.output_sockets["output"] = min(embedding1, embedding2)

    def to_dict(self) -> dict:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> 'MinNode':
        return cls(data["id"])

    @classmethod
    def get_required_pipelines(cls) -> Set[str]:
        return set()


class MaxNode(Node):
    def __init__(self, node_id: str):
        super().__init__(
            node_id,
            input_sockets={"embedding1", "embedding2"},
            output_sockets={"output"}
        )

    def process(self, PipelineRegistery) -> None:
        embedding1 = self.input_sockets["embedding1"]
        embedding2 = self.input_sockets["embedding2"]

        self.output_sockets["output"] = max(embedding1, embedding2)

    def to_dict(self) -> dict:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> 'MaxNode':
        return cls(data["id"])

    @classmethod
    def get_required_pipelines(cls) -> Set[str]:
        return set()


class PairwiseMergeNode(Node):
    def __init__(self, node_id: str):
        super().__init__(
            node_id,
            input_sockets={"embedding1", "embedding2"},
            output_sockets={"output"}
        )

    def process(self, PipelineRegistery) -> None:
        # Ensure all input sockets are connected
        for inputSock in self.input_sockets.keys():
            if self.input_sockets[inputSock] is None:
                raise ValueError(f"Input socket ({inputSock}) not connected")

        # Retrieve embeddings
        embedding1 = self.input_sockets["embedding1"]
        embedding2 = self.input_sockets["embedding2"]

        # Perform pairwise merge (median-like blend)
        stacked = np.stack((embedding1, embedding2), axis=0)
        sorted_stack = np.sort(stacked, axis=0)
        pairwise_merge = sorted_stack[len(stacked) // 2]

        # Store the result in the output socket
        self.output_sockets["output"] = pairwise_merge

    def to_dict(self) -> dict:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> 'PairwiseMergeNode':
        return cls(data["id"])

    @classmethod
    def get_required_pipelines(cls) -> Set[str]:
        return set()


# GRAPH, GRAPH EXECUTOR, ETC
class Graph:
    NODE_TYPES = {
        'LoadEmbeddingNode': LoadEmbeddingNode,
        'SaveEmbeddingNode': SaveEmbeddingNode,
        'InferenceNode': InferenceNode,
        'NormalizeNode': NormalizeNode,
        'WeightedAverageNode': WeightedAverageNode,
        'AdditionNode': AdditionNode,
        'SubtractionNode': SubtractionNode,
        'MultiplicationNode': MultiplicationNode,
        'DivisionNode': DivisionNode,
        'MeanNode': MeanNode,
        'MinNode': MinNode,
        'MaxNode': MaxNode,
        'PairwiseMergeNode': PairwiseMergeNode,
        'StyleTransferNode': StyleTransferNode
    }

    def __init__(self, pipeline_registery):
        self.nodes: Dict[str, Node] = {}
        self.connections: List[Connection] = []
        self.pipeline_registery = pipeline_registery

    def validate_pipeline_requirements(self) -> List[str]:
        """Check if all required pipeline classes are registered"""
        missing_pipelines = set()
        for node in self.nodes.values():
            required = node.get_required_pipelines()
            for pipeline_name in required:
                if PipelineRegistry.get(pipeline_name) is None:
                    missing_pipelines.add(pipeline_name)
        return list(missing_pipelines)

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def remove_node(self, node: Node) -> None:
        del self.nodes[node.id]

    def add_connection(self, connection: Connection) -> None:
        self.connections.append(connection)

    def remove_connection(self, connection: Connection) -> None:
        for con in self.connections:
            if con.to_dict() == connection.to_dict():
                self.connections.remove(con)

    def process(self) -> None:
        # Validate pipeline requirements before processing
        missing = self.validate_pipeline_requirements()
        if missing:
            raise RuntimeError(
                f"Missing required pipeline classes: {', '.join(missing)}. "
                "Please register them using PipelineRegistry.register() before processing."
            )

        processed = set()

        def process_node(node_id: str) -> None:
            print(f"Processing node: {type(self.nodes[node_id]).__name__} - {node_id}")
            if node_id in processed:
                return

            for conn in self.connections:
                if conn.to_node == node_id and conn.from_node not in processed:
                    process_node(conn.from_node)

            for conn in self.connections:
                if conn.to_node == node_id:
                    from_node = self.nodes[conn.from_node]
                    to_node = self.nodes[conn.to_node]
                    value = from_node.get_output(conn.from_socket)
                    to_node.set_input(conn.to_socket, value)

            self.nodes[node_id].process(self.pipeline_registery)
            processed.add(node_id)

        for node_id in self.nodes:
            process_node(node_id)

    def save_to_json(self, filepath: str) -> None:
        # Warn about pipeline dependencies
        required_pipelines = set()
        for node in self.nodes.values():
            required_pipelines.update(node.get_required_pipelines())

        if required_pipelines:
            warnings.warn(
                f"This graph requires the following pipeline classes: {', '.join(required_pipelines)}. "
                "Make sure to register them using PipelineRegistry.register() before loading the graph.",
                UserWarning
            )

        node_dict = {}

        data = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "connections": [conn.to_dict() for conn in self.connections],
            "required_pipelines": list(required_pipelines),
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str, pipeline_registery) -> 'Graph':
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Check pipeline requirements
        required_pipelines = set(data.get("required_pipelines", []))
        missing_pipelines = [p for p in required_pipelines if pipeline_registery.get(p) is None]

        if missing_pipelines:
            raise RuntimeError(
                f"Missing required pipeline classes: {', '.join(missing_pipelines)}. "
                "Please register them using PipelineRegistry.register() before loading the graph."
            )

        graph = cls(pipeline_registery=pipeline_registery)

        # Create nodes
        for node_data in data["nodes"]:
            node_type = node_data["type"]
            if node_type not in cls.NODE_TYPES:
                raise ValueError(f"Unknown node type: {node_type}")

            node_class = cls.NODE_TYPES[node_type]
            node = node_class.from_dict(node_data)
            graph.add_node(node)

        # Create connections
        for conn_data in data["connections"]:
            connection = Connection.from_dict(conn_data)
            graph.add_connection(connection)

        return graph


# EXAMPLE
def example_usage():
    # Register the pipeline class
    register = PipelineRegistry()
    register.register("StyleTTSPipeline", "STYLETTS2PIPELINE HERE")

    # Create graph
    graph = Graph(register)

    # Create nodes
    load_node = LoadEmbeddingNode("load1", "embedding1.mp3", "StyleTTSPipeline")
    normalize_node = NormalizeNode("norm1")

    # Add nodes to graph
    graph.add_node(load_node)
    graph.add_node(normalize_node)

    # Create connection
    connection = Connection("load1", "embedding", "norm1", "embedding")
    graph.add_connection(connection)

    # Save graph
    graph.save_to_json("example_graph.json")

    loaded_graph = Graph.load_from_json("example_graph.json", pipeline_registery=register)
    # loaded_graph.process()
