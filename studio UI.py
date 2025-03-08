import inspect

import dearpygui.dearpygui as dpg
from typing import Dict, Optional, List, Set, Any
import json
import numpy as np
import os
from STTS2Dash import tts, studio

# Initialize DearPyGUI
dpg.create_context()


class NodeEditor:
    def __init__(self, pipeline_registry):
        self.pipeline_registry = pipeline_registry
        self.graph = studio.Graph(pipeline_registry)
        self.node_positions: Dict[str, tuple] = {}
        self.next_node_id = 1
        self.setup_gui()

    def setup_gui(self):
        # Create window and main menu
        dpg.create_viewport(title="STTS-Dash GraphUI", width=1280, height=720)

        with dpg.window(label="Node Editor", tag="main_window"):
            with dpg.menu_bar():
                with dpg.menu(label="File"):
                    dpg.add_menu_item(label="New", callback=self.new_graph)
                    dpg.add_menu_item(label="Save", callback=self.save_graph)
                    dpg.add_menu_item(label="Load", callback=self.load_graph)
                    dpg.add_menu_item(label="Run Graph", callback=self.run_graph)

                with dpg.menu(label="Add Node"):
                    for node_type in self.graph.NODE_TYPES.keys():
                        dpg.add_menu_item(label=node_type, callback=self.add_node, user_data=node_type)

            # Node editor
            with dpg.node_editor(tag="node_editor",
                                 callback=self.link_callback,
                                 width=1280,
                                 height=640,
                                 minimap=True,
                                 minimap_location=dpg.mvNodeMiniMap_Location_TopLeft):
                pass

        # Add context menu for nodes
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=self.show_context_menu)

        # Create context menu
        with dpg.window(label="Context Menu", show=False, tag="node_context_menu", no_title_bar=True,
                        autosize=True):
            dpg.add_menu_item(label="Delete Node", callback=self.delete_selected_node)

        # File dialog
        with dpg.file_dialog(
                directory_selector=False, show=False,
                callback=self.file_dialog_callback,
                tag="file_dialog",
                width=700, height=400
        ):
            dpg.add_file_extension(".json")

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def generate_unique_id(self) -> str:
        """Generate a unique node ID"""
        node_id = f"node-{self.next_node_id}"
        self.next_node_id += 1
        return node_id

    def add_node(self, sender, app_data, node_type: str):
        """Add a new node to the graph and UI"""
        node_id = self.generate_unique_id()

        # Get node class from Graph.NODE_TYPES
        node_class = self.graph.NODE_TYPES[node_type]

        # Inspect the constructor parameters
        params = inspect.signature(node_class.__init__).parameters

        # Create constructor arguments
        constructor_args = {'node_id': node_id}

        # Add default values for other parameters
        for param_name, param in params.items():
            if param_name not in ['self', 'node_id']:
                if param_name == 'pipeline_name':
                    constructor_args[param_name] = 'StyleTTSPipeline'
                elif param.default != inspect.Parameter.empty:
                    constructor_args[param_name] = param.default
                elif param.annotation == str:
                    constructor_args[param_name] = ""
                elif param.annotation == float:
                    constructor_args[param_name] = 0.0
                elif param.annotation == int:
                    constructor_args[param_name] = 0

        # Create node instance
        node = node_class(**constructor_args)
        self.graph.add_node(node)

        def create_callback(node_ref, param_name):
            """Create a callback that updates the node's parameter"""

            def callback(sender, app_data):
                setattr(node_ref, param_name, app_data)
                # Update the node in the graph
                self.graph.nodes[node_ref.id] = node_ref

            return callback

        # Create node in UI
        print(f"added {node_type}: {node_id}")

        with dpg.node(label=node_type, tag=node_id, parent="node_editor"):
            # Add input sockets
            for socket_name in node._input_socket_names:
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, tag=f"{node_id}_{socket_name}"):
                    dpg.add_text(socket_name.capitalize())

            # Add parameters
            if len(params) > 2:  # More than just self and node_id
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static, tag=f"{node_id}_params"):
                    for param_name, param in params.items():
                        if param_name not in ['self', 'node_id', 'input_sockets', 'output_sockets', 'pipeline_name']:
                            value = getattr(node, param_name)

                            # Create appropriate UI control based on parameter type
                            if param.annotation == str:
                                dpg.add_input_text(
                                    label=param_name.replace('_', ' ').title(),
                                    default_value=value,
                                    user_data=param_name,
                                    callback=create_callback(node, param_name), width=150
                                )
                            elif param.annotation == float:
                                # Determine appropriate range based on parameter name
                                if 'scale' in param_name:
                                    min_val, max_val = 0.0, 5.0
                                elif param_name in ['alpha', 'beta']:
                                    min_val, max_val = 0.0, 1.0
                                elif param_name == 'speed':
                                    min_val, max_val = 0.5, 2.0
                                elif 'weight' in param_name:
                                    min_val, max_val = 0.0, 100.0
                                else:
                                    min_val, max_val = 0.0, 100.0

                                dpg.add_slider_float(
                                    label=param_name.replace('_', ' ').title(),
                                    default_value=value,
                                    min_value=min_val,
                                    max_value=max_val,
                                    callback=create_callback(node, param_name),
                                    user_data=param_name, width=150
                                )
                            elif param.annotation == int:
                                # Determine appropriate range based on parameter name
                                if 'steps' in param_name:
                                    min_val, max_val = 1, 50
                                else:
                                    min_val, max_val = 0, 100

                                dpg.add_slider_int(
                                    label=param_name.replace('_', ' ').title(),
                                    default_value=value,
                                    min_value=min_val,
                                    max_value=max_val,
                                    callback=create_callback(node, param_name),
                                    user_data=param_name, width=150
                                )

            # Add output sockets
            for socket_name in node._output_socket_names:
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag=f"{node_id}_{socket_name}"):
                    dpg.add_text(socket_name.capitalize())

    def show_context_menu(self, sender, app_data):
        if dpg.is_item_hovered("node_editor"):
            mouse_pos = dpg.get_mouse_pos()

            # Get the item under the mouse
            selected_node = None
            for node_id in self.graph.nodes.keys():
                if dpg.is_item_hovered(node_id):
                    selected_node = node_id
                    break

            if selected_node:
                dpg.set_item_user_data("node_context_menu", selected_node)
                dpg.configure_item("node_context_menu", show=True, pos=mouse_pos)

    def delete_selected_node(self):
        node_id = dpg.get_item_user_data("node_context_menu")
        self.delete_node(node_id)
        dpg.configure_item("node_context_menu", show=False)

    def delete_node(self, node_id: str):
        # Remove all connections involving this node
        connections_to_remove = []
        for conn in self.graph.connections:
            if conn.from_node == node_id or conn.to_node == node_id:
                connections_to_remove.append(conn)

                # Remove the visual link
                try:
                    link_id = dpg.get_alias_id(f"{conn.from_node}_{conn.from_socket}###{conn.to_node}_{conn.to_socket}")
                    dpg.delete_item(link_id)
                except Exception as e:
                    print(e)

        # Remove connections from graph
        for conn in connections_to_remove:
            if hasattr(self.graph, 'remove_connection'):
                self.graph.remove_connection(conn)
            else:
                self.graph.connections = [c for c in self.graph.connections if c != conn]

        # Remove node from graph
        if node_id in self.graph.nodes:
            del self.graph.nodes[node_id]

        # Remove node from UI
        dpg.delete_item(node_id)

    def link_callback(self, sender, app_data):
        # app_data contains: (link_id, from_attr, to_attr)
        from_node = dpg.get_item_parent(app_data[0])
        to_node = dpg.get_item_parent(app_data[1])

        from_id = dpg.get_item_alias(app_data[0]).split('_')[0]
        to_id = dpg.get_item_alias(app_data[1]).split('_')[0]

        print(f"{dpg.get_item_alias(app_data[0])} >> {dpg.get_item_alias(app_data[1])}")

        # Get socket names from attribute tags
        from_socket = dpg.get_item_alias(app_data[0]).split('_')[-1]
        to_socket = dpg.get_item_alias(app_data[1]).split('_')[-1]

        dpg.add_node_link(app_data[0], app_data[1], parent=sender)

        # Create connection in graph
        connection = studio.Connection(str(from_id), from_socket, str(to_id), to_socket)
        self.graph.add_connection(connection)

        # Update the receiving node's input socket
        to_node = self.graph.nodes[str(to_id)]
        from_node = self.graph.nodes[str(from_id)]
        to_node.set_input(to_socket, from_node.get_output(from_socket))

    def new_graph(self):
        """Create a new empty graph"""
        self.graph = studio.Graph(self.pipeline_registry)
        dpg.delete_item("node_editor", children_only=True)
        self.next_node_id = 1

    def save_graph(self):
        """Show save file dialog"""
        dpg.show_item("file_dialog")
        self.file_dialog_mode = "save"

    def load_graph(self):
        """Show load file dialog"""
        dpg.show_item("file_dialog")
        self.file_dialog_mode = "load"

    def file_dialog_callback(self, sender, app_data):
        """Handle file dialog results"""
        file_path = app_data['file_path_name']

        if self.file_dialog_mode == "save":
            self.graph.save_to_json(file_path)
        else:  # load
            self.new_graph()  # Clear current graph
            self.graph = studio.Graph.load_from_json(file_path, self.pipeline_registry)
            self.recreate_nodes_from_graph()
            self.next_node_id = len(self.graph.nodes) + 1

    def recreate_nodes_from_graph(self):
        """Recreate UI nodes from loaded graph"""
        # Create nodes
        for node_id, node in self.graph.nodes.items():
            node_type = node.__class__.__name__

            # Get node class from Graph.NODE_TYPES
            node_class = self.graph.NODE_TYPES[node_type]

            # Inspect the constructor parameters
            params = inspect.signature(node_class.__init__).parameters

            def create_callback(node_ref, param_name):
                """Create a callback that updates the node's parameter"""

                def callback(sender, app_data):
                    setattr(node_ref, param_name, app_data)
                    # Update the node in the graph
                    self.graph.nodes[node_ref.id] = node_ref

                return callback

            # Create node in UI
            with dpg.node(label=node_type, tag=node_id, parent="node_editor"):
                # Add input sockets
                for socket_name in node._input_socket_names:
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, tag=f"{node_id}_{socket_name}"):
                        dpg.add_text(socket_name.capitalize())

                # Add parameters
                if len(params) > 2:  # More than just self and node_id
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static, tag=f"{node_id}_params"):
                        for param_name, param in params.items():
                            if param_name not in ['self', 'node_id', 'input_sockets', 'output_sockets', 'pipeline_name']:
                                value = getattr(node, param_name)

                                # Create appropriate UI control based on parameter type
                                if param.annotation == str:
                                    dpg.add_input_text(
                                        label=param_name.replace('_', ' ').title(),
                                        default_value=value,
                                        callback=create_callback(node, param_name),
                                        user_data=param_name, width=150
                                    )
                                elif param.annotation == float:
                                    # Determine appropriate range based on parameter name
                                    if 'scale' in param_name:
                                        min_val, max_val = 0.0, 2.0
                                    elif param_name in ['alpha', 'beta']:
                                        min_val, max_val = 0.0, 1.0
                                    elif param_name == 'speed':
                                        min_val, max_val = 0.5, 2.0
                                    elif 'weight' in param_name:
                                        min_val, max_val = 0.0, 100.0
                                    else:
                                        min_val, max_val = 0.0, 100.0

                                    dpg.add_slider_float(
                                        label=param_name.replace('_', ' ').title(),
                                        default_value=value,
                                        min_value=min_val,
                                        max_value=max_val,
                                        callback=create_callback(node, param_name),
                                        user_data=param_name, width=150
                                    )
                                elif param.annotation == int:
                                    # Determine appropriate range based on parameter name
                                    if 'steps' in param_name:
                                        min_val, max_val = 1, 50
                                    else:
                                        min_val, max_val = 0, 100

                                    dpg.add_slider_int(
                                        label=param_name.replace('_', ' ').title(),
                                        default_value=value,
                                        min_value=min_val,
                                        max_value=max_val,
                                        callback=create_callback(node, param_name),
                                        user_data=param_name, width=150
                                    )

                # Add output sockets
                for socket_name in node._output_socket_names:
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag=f"{node_id}_{socket_name}"):
                        dpg.add_text(socket_name.capitalize())

        # Recreate connections
        for conn in self.graph.connections:
            dpg.add_node_link(
                f"{conn.from_node}_{conn.from_socket}",
                f"{conn.to_node}_{conn.to_socket}",
                parent="node_editor"
            )

    def run_graph(self):
        """Execute the current graph"""
        try:
            self.graph.process()
            print("Graph executed successfully!")
        except Exception as e:
            print(f"Error executing graph: {str(e)}")

    def run(self):
        """Start the application"""
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

        dpg.destroy_context()


# Example usage
if __name__ == "__main__":
    # Create pipeline registry and register StyleTTS pipeline
    ttsPipe = tts.StyleTTS2Pipeline()
    ttsPipe.load_from_files("./test_models/....pth",
                            "./test_models/....yml", is_tsukasa=True)

    register = studio.PipelineRegistry()
    register.register("StyleTTSPipeline", ttsPipe)

    # Create and run editor
    editor = NodeEditor(register)
    editor.run()
