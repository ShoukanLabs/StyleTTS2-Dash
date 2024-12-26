from STTS2Dash import tts, studio

# Register the pipeline class
ttsPipe = tts.StyleTTS2Pipeline()
ttsPipe.load_from_files("./test_models/epoch 7.pth",
                        "./test_models/config_n.yml", is_tsukasa=True)

register = studio.PipelineRegistry()
register.register("StyleTTSPipeline", ttsPipe)

# Create graph
editor = studio.editor(register)

# graph = studio.Graph(register)
#
# # Create nodes
# load_node = studio.LoadEmbeddingNode("load1", "./test_models/audio.wav", "StyleTTSPipeline")
# load_node2 = studio.LoadEmbeddingNode("load2", "test_models/n-sample.wav", "StyleTTSPipeline")
# load_node3 = studio.LoadEmbeddingNode("load3", "test_models/expressive.mp3", "StyleTTSPipeline")
# average = studio.WeightedAverageNode("avg", weight=40)
# average2 = studio.WeightedAverageNode("avg2", weight=50)
# test = studio.InferenceNode("out",
#                                 "I have high standards for myself, and i think it's only right for these standards to apply to you as well.",
#                                 "./test.wav", "StyleTTSPipeline", embedding_scale=2, alpha=0.5, beta=0.6, speed=1.2)
#
# # Add nodes to graph
# graph.add_node(load_node)
# graph.add_node(load_node2)
# graph.add_node(load_node3)
# graph.add_node(average)
# graph.add_node(average2)
# graph.add_node(test)
#
# # Create connection
# connection = studio.Connection("load1", "embedding", "avg", "embedding1")
# connection2 = studio.Connection("load2", "embedding", "avg", "embedding2")
# connection3 = studio.Connection("load3", "embedding", "avg2", "embedding1")
# connection4 = studio.Connection("avg", "output", "avg2", "embedding2")
# avg = studio.Connection("avg2", "output", "out", "embedding")
# graph.add_connection(connection)
# graph.add_connection(connection2)
# graph.add_connection(connection3)
# graph.add_connection(connection4)
# graph.add_connection(avg)
#
# graph.process()
#
# # Save graph
# graph.save_to_json("example_graph.json")
