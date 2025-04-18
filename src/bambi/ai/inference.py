import abc


class Inference(abc.ABC):
    """
    Abstract class representing a AI model inference
    """
    @abc.abstractmethod
    def apply(self, input_path: str, output_path: str) -> None:
        """
        Method for applying the model to the given input, creating the given output
        :param input_path: Input that should be analysed with AI inference
        :param output_path: Output created by AI model
        :return:
        """
        pass

model_registry = {}