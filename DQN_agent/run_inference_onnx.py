import numpy as np
import onnxruntime as ort

def run_inference_onnx(onnx_model_path, input_data):
    """
    Laad het ONNX-model en doe inference met onnxruntime.
    Er is geen PyTorch nodig.
    """
    # Initialiseer een onnxruntime sessie
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    # Haal de naam van de eerste (enige) input op
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Voer inference uit
    outputs = session.run([output_name], {input_name: input_data})
    # outputs is een lijst met arrays, in dit geval [ (batch_size, 81) ]
    return outputs[0]  # De numpy array met Q-waarden

if __name__ == "__main__":
    # 1. Geef het pad van je geëxporteerde ONNX-model
    onnx_path = "team35_9x9_dqn_model.onnx"

    # 2. Maak een dummy input: (batch_size=1, 3, 9, 9)
    # of je eigen board data als je live inference doet
    test_input = np.random.randn(1, 3, 9, 9).astype(np.float32)

    # 3. Voer inference uit
    q_values = run_inference_onnx(onnx_path, test_input)
    print("Output shape:", q_values.shape)  # (1, 81)
    print("Voorbeeld Q-waarden:", q_values[0, :10])
