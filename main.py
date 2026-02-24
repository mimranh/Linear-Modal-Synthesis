from load_audio import load_audio
from modal_analysis import ModalFeatureExtractor
from cross_resolution import CrossResolutionValidator
from diagnostics import ModalDiagnostics

signal, fs = load_audio("wood_door.wav")

extractor = ModalFeatureExtractor(fs)
features_per_resolution = extractor.extract_features(signal)
for i, res in enumerate(features_per_resolution):
    print(f"Resolution {i}: {len(res)} modes")

validator = CrossResolutionValidator()
clusters = validator.cluster_modes(features_per_resolution)
validated_modes = validator.validate_clusters(clusters)

diag = ModalDiagnostics()
diag.plot_mode_statistics(validated_modes)

print("Number of validated modes:", len(validated_modes))


