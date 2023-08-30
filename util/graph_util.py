###############################################################################
#
# Exxfer Graph Utility 
#
# Copyright (C) 2023 EXXFER Research Project
# Author: Dominik Neumann <dominik.neumann@exxeta.com>
#
# URL: <https://exxfer.io/>
###############################################################################

###############################################################################
# PYTHON_2 model: There is just an Entity Model
# ------------------------- entity keys    --------------------------------------
# ENTITY = ['@type', 'comments', 'javaDoc', 'complexity', 'entropy', 'entityFullQualifiedName', 
# 'entityId', 'entityName', 'entityOrigin', 'entityType', 'interfaces', 'language', 
#'loc', 'qloc', 'visibility', 'package', 'fields', 'annotation_fields', 'methods', 
# 'modifiers', 'imports', 'extends', 'classLiterals', 'filename', 'encoding', 'annotations', 
# 'is_entrypoint', 'entrypoint_reason', 'is_inner', 'touched_by_ep', 'additional_attributes', 
# 'modularized'] 
# 
# JAVA model: There are two models: Entities and Elements
# ------------------------- entity keys    --------------------------------------
# ENTITY = ['id', 'comments', 'endLine', 'entityFullQualifiedName', 'entityId', 
# 'entityName', 'entityOrigin', 'entityType', 'entrypoint', 'exitpoint', 'javaDoc', 
# 'language', 'loc', 'package', 'qloc', 'startLine', 'visibility']
# ------------------------- entity relation keys  -------------------------------
# RELATION = ['source', 'target', 'from', 'relationId', 'relationType', 'to']
#
# ------------------------- element keys    --------------------------------------
# ELEMENT =  ['id', 'elementFullQualifiedName', 'elementId', 'elementName', 'elementType', 
# 'entrypoint', 'exitpoint', 'parentEntityFullQualifiedName', 'parentEntityOrigin']
# ------------------------- relation keys  --------------------------------------
# RELATION = ['source', 'target', 'from', 'invocationId', 'invocationType', 'to']
#
###############################################################################
import networkx as nx
import numpy as np

from typing import List, Set
from util.graph_model import EXXFER_MODEL_JAVA, EXXFER_MODEL_PYTHON_2
from util.graph_model import ExxferJava, ExxferPython2

########################## static function independend of EXXFER model version
def get_clusters(G, clustering: str):
    '''returns a dict object containig node list for each cluster

    inputs:
        G graph
        clustering defines the node attribute in the graph that indicates the cluster
    
    output:
        dictionary of clusters with nodes belonging to a cluster
        {cluster_name : list(nodes)}
    '''
    nodes = G.nodes
    clusters = {} 
    for node in nodes:
        cluster = nodes.get(node).get(clustering, None) 
        #if cluster is not None:
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(node)
    return clusters

def get_clusters_for_entities(entities: List[dict], clustering: str):
    '''returns a dict object containig entity list for each cluster

    inputs:
        list of entities
        clustering defines the entity attribute indicating the cluster

    output:
        dictionary of clusters with entities belonging to a cluster
        {cluster_name : list(entities)}
    '''
    clusters = {} 
    for entity in entities:
        cluster = entity.get(clustering, None) 
        #if cluster is not None:
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(entity)
    return clusters

def get_clusters(G, clustering_type: str, cluster_starts_with=None):
    '''returns a dict object containig a node list for each cluster

    inputs:
        G graph
        clustering_type defines the node attribute in the graph that 
        indicates the cluster
        cluster_starts_with is a list of filter. if not None then only clusters 
        are considered that starts with...
    
    output:
        dictionary of clusters with nodes belonging to a cluster
        {cluster_name : list(nodes)}
    '''
    nodes = G.nodes
    clusters = {} 
    for node in nodes:
        cluster = nodes.get(node).get(clustering_type, None) 
        if cluster:
            filter_match = False
            if cluster_starts_with is not None and len(cluster_starts_with) > 0:
                for _filter in cluster_starts_with:
                    if cluster.startswith(_filter):
                        filter_match = True
                        break
            else:
                filter_match = True
            if filter_match:
                if cluster not in clusters:
                    clusters[cluster] = []
                node_list = clusters.get(cluster)
                node_list.append(node)
                clusters[cluster] = node_list
    return clusters

def get_clusters_by_cluster_level(G, clustering_type: str, level: int):
    nodes = G.nodes
    clusters = {} 
    return clusters

def node_list_similarity(node_list_1: list, node_list_2: list):
    '''returns the similarity of two node lists. same result as Jaccard Index'''
    J = -1
    A = set(node_list_1)
    B = set(node_list_2)
    A_i_B = A.intersection(B)
    A_u_B = A.union(B)
    if len(A_u_B) != 0:
        J = len(A_i_B) / len(A_u_B)
    return J

def compare_clusterings(clustering_i: dict, clustering_j: dict):
    '''returns the similarity matrix of two clusterings'''
    n = len(clustering_i)
    m = len(clustering_j)
    SIM = np.zeros((n,m))
    clusters_ids_1 = {}
    clusters_ids_2 = {}
    for i, cluster_i in enumerate(clustering_i):
        node_list_i = clustering_i[cluster_i]
        clusters_ids_1[cluster_i] = i
        for j, cluster_j in enumerate(clustering_j):
            node_list_j = clustering_j[cluster_j]
            clusters_ids_2[cluster_j] = j
            SIM[i,j] = node_list_similarity(node_list_i,node_list_j)

    return SIM, clusters_ids_1, clusters_ids_2

###############################################################################
#
# Utilitiy for analyzing EXXFER scans
#
###############################################################################
class GraphUtil():

    def __init__(self, exxfer_model):

        self.exxfer_model = exxfer_model
        self.set_constants()

    ###########################################################################
    def get_entities_by_type(self, entities: List[dict]) -> dict:
        entities_by_type = {}
        for entity in entities:
            if entity[self.ENTITY_TYPE] not in entities_by_type:
                entities_by_type[entity[self.ENTITY_TYPE]] = []
            entities_by_type[entity[self.ENTITY_TYPE]].append(entity)
        return dict(sorted(entities_by_type.items() , key=lambda x: len(x[1]), reverse=True))

    def get_elements_by_type(self, elements: List[dict]) -> dict:
        elements_by_type = {}
        for entity in elements:
            if entity[self.ELEMENT_TYPE] not in elements_by_type:
                elements_by_type[entity[self.ELEMENT_TYPE]] = []
            elements_by_type[entity[self.ELEMENT_TYPE]].append(entity)
        return dict(sorted(elements_by_type.items() , key=lambda x: len(x[1]), reverse=True))

    def get_entities_by_type_and_origin(self, entities: List[dict], sorted_by="type") -> dict:
        entities_by_type_origin = {}
        for entity in entities:

            key = (entity[self.ENTITY_TYPE], entity[self.ENTITY_ORIGIN])
            
            if key not in entities_by_type_origin:
                entities_by_type_origin[key] = []
            entities_by_type_origin[key].append(entity)

        sort_keys = [(_type, _origin, len(items)) for (_type, _origin), items in entities_by_type_origin.items()]

        if sorted_by == "type":
            sort_keys = sorted(sort_keys, key=lambda x: x[0], reverse=True)
        elif sorted_by == "origin":
            sort_keys = sorted(sort_keys, key=lambda x: x[1], reverse=True)
        elif sorted_by == "size":
            sort_keys = sorted(sort_keys, key=lambda x: x[2], reverse=True)

        entities_by_type_origin_sorted = {}
        for (_type, _origin, _) in sort_keys:
            key = (_type, _origin)
            entities_by_type_origin_sorted[key] = entities_by_type_origin[key]

        return entities_by_type_origin_sorted

    def get_entity_relations_by_type(self, relations: List[dict], allowed_entities=None) -> dict:
        allowed_entity_ids = []
        if allowed_entities is not None: 
            allowed_entity_ids = [entity[self.ENTITY_ID] for entity in allowed_entities]
        relations_by_type = {}
        for relation in relations:
            _add = True
            if allowed_entities is not None:
                if relation[self.ENTITY_RELATION_FROM] not in allowed_entity_ids:
                    _add = False

                if relation[self.ENTITY_RELATION_TO] not in allowed_entity_ids:
                    _add = False           
            if _add:
                if relation[self.ENTITY_RELATION_TYPE] not in relations_by_type:
                    relations_by_type[relation[self.ENTITY_RELATION_TYPE]] = []
                relations_by_type[relation[self.ENTITY_RELATION_TYPE]].append(relation)
        return dict(sorted(relations_by_type.items() , key=lambda x: len(x[1]), reverse=True))

    def get_element_relations_by_type(self, relations: List[dict]) -> dict:
        relations_by_type = {}
        for relation in relations:
            if relation[self.ELEMENT_RELATION_TYPE] not in relations_by_type:
                relations_by_type[relation[self.ELEMENT_RELATION_TYPE]] = []
            relations_by_type[relation[self.ELEMENT_RELATION_TYPE]].append(relation)
        return dict(sorted(relations_by_type.items() , key=lambda x: len(x[1]), reverse=True))

    ###########################################################################

    def set_constants(self):

        if self.exxfer_model == EXXFER_MODEL_JAVA:
            model = ExxferJava()
        elif self.exxfer_model == EXXFER_MODEL_PYTHON_2:
            model = ExxferPython2()
        else:
            raise RuntimeError(f"Could not initialize for exxfer model = {self.exxfer_model}")

        self.ELEMENT_FQN = model.ELEMENT_FQN
        self.ELEMENT_ID = model.ELEMENT_ID
        self.ELEMENT_TYPE = model.ELEMENT_TYPE
        self.ELEMENT_IS_EP = model.ELEMENT_IS_EP
        self.ELEMENT_IS_EX = model.ELEMENT_IS_EX 
        self.ELEMENT_ORIGIN = model.ELEMENT_ORIGIN       
        self.ELEMENT_ORIGIN_ID = model.ELEMENT_ORIGIN_ID
        self.ELEMENT_ORIGIN_FQN = model.ELEMENT_ORIGIN_FQN
        
        self.ELEMENT_RELATION_TYPE = model.ELEMENT_RELATION_TYPE
        self.ELEMENT_RELATION_FROM = model.ELEMENT_RELATION_FROM
        self.ELEMENT_RELATION_TO = model.ELEMENT_RELATION_TO
        self.ELEMENT_RELATION_ID = model.ELEMENT_RELATION_ID

        self.ELEMENT_RELATION_TYPE_METHOD_TO_METHOD = model.ELEMENT_RELATION_TYPE_METHOD_TO_METHOD
        self.ELEMENT_RELATION_TYPE_FIELD_TO_METHOD = model.ELEMENT_RELATION_TYPE_FIELD_TO_METHOD
        self.ELEMENT_RELATION_TYPE_INITIALIZER_TO_METHOD = model.ELEMENT_RELATION_TYPE_INITIALIZER_TO_METHOD

        self.ELEMENT_TYPE_METHOD = model.ELEMENT_TYPE_METHOD
        self.ELEMENT_TYPE_FIELD = model.ELEMENT_TYPE_FIELD
        self.ELEMENT_TYPE_INITIALIZER = model.ELEMENT_TYPE_INITIALIZER

        self.ENTITY_RELATION_TYPE_METHODPARAMETER = model.ENTITY_RELATION_TYPE_METHODPARAMETER
        self.ENTITY_RELATION_TYPE_METHODFIELD = model.ENTITY_RELATION_TYPE_METHODFIELD
        self.ENTITY_RELATION_TYPE_METHODRETURN = model.ENTITY_RELATION_TYPE_METHODRETURN
        self.ENTITY_RELATION_TYPE_METHODEXCEPTION = model.ENTITY_RELATION_TYPE_METHODEXCEPTION
        self.ENTITY_RELATION_TYPE_METHODUSEDTYPE = model.ENTITY_RELATION_TYPE_METHODUSEDTYPE
        self.ENTITY_RELATION_TYPE_INVOCATIONRETURN = model.ENTITY_RELATION_TYPE_INVOCATIONRETURN
        self.ENTITY_RELATION_TYPE_METHODVARIABLE = model.ENTITY_RELATION_TYPE_METHODVARIABLE
        self.ENTITY_RELATION_TYPE_ORM = model.ENTITY_RELATION_TYPE_ORM
        self.ENTITY_RELATION_TYPE_CRUD = model.ENTITY_RELATION_TYPE_CRUD
        self.ENTITY_RELATION_TYPE_INHERITANCE = model.ENTITY_RELATION_TYPE_INHERITANCE
        self.ENTITY_RELATION_TYPE_INTERFACE = model.ENTITY_RELATION_TYPE_INTERFACE
        self.ENTITY_RELATION_TYPE_ANNOTATION = model.ENTITY_RELATION_TYPE_ANNOTATION
        self.ENTITY_RELATION_TYPE_ANNOTATION_VALUE = model.ENTITY_RELATION_TYPE_ANNOTATION_VALUE
        self.ENTITY_RELATION_TYPE_FIELD = model.ENTITY_RELATION_TYPE_FIELD
        self.ENTITY_RELATION_TYPE_CLASSLITERAL = model.ENTITY_RELATION_TYPE_CLASSLITERAL

        self.ENTITY_TYPE = model.ENTITY_TYPE
        self.ENTITY_ID = model.ENTITY_ID
        self.ENTITY_PACKAGE = model.ENTITY_PACKAGE
        self.ENTITY_NAME = model.ENTITY_NAME
        self.ENTITY_FQN = model.ENTITY_FQN
        self.ENTITY_METHODS = model.ENTITY_METHODS
        self.ENTITY_IS_EP = model.ENTITY_IS_EP
        self.ENTITY_IS_EX = model.ENTITY_IS_EX
        self.ENTITY_ORIGIN = model.ENTITY_ORIGIN
    
        self.ENTITY_METHOD_FQN =model.ENTITY_METHOD_FQN
        self.ENTITY_METHOD_PARAMETERS = model.ENTITY_METHOD_PARAMETERS
        self.ENTITY_METHOD_PARAMETER_TYPE = model.ENTITY_METHOD_PARAMETER_TYPE
        self.ENTITY_METHOD_FIELDS = model.ENTITY_METHOD_FIELDS
        self.ENTITY_METHOD_INVOCATIONS = model.ENTITY_METHOD_INVOCATIONS
        self.ENTITY_METHOD_IS_EP =model.ENTITY_METHOD_IS_EP

        self.ENTITY_RELATION_TYPE = model.ENTITY_RELATION_TYPE
        self.ENTITY_RELATION_FROM = model.ENTITY_RELATION_FROM
        self.ENTITY_RELATION_TO = model.ENTITY_RELATION_TO
        self.ENTITY_RELATION_METHOD = model.ENTITY_RELATION_METHOD
        self.ENTITY_RELATION_ID = model.ENTITY_RELATION_ID

        self.ENTITY_RELATION_TYPE_CALL = model.ENTITY_RELATION_TYPE_CALL
        self.ENTITY_RELATION_TYPE_INNERCLASS = model.ENTITY_RELATION_TYPE_INNERCLASS


    ###########################################################################
    #
    # GraphBuilder - Part
    #
    # Program Dependency Graph
    # Call Graph
    # Bipartite Graph 
    #
    ###########################################################################

    ###############  Build the program dependency graph #######################
    def build_program_dependency_graph(self,
                                    entities: List[dict],
                                    relations: List[dict],
                                    allowed_entity_types: List[str]=[],
                                    allowed_relation_types: List[str]=[], 
                                    entity_attributes: List[str]=None,
                                    directed: bool=True, 
                                    isolated_nodes: bool=True,
                                    self_loops: bool=True,
                                    weight: str=None,
                                    clustering: str=None,
                                    multi_edges=False) -> nx.Graph | nx.DiGraph | nx.MultiDiGraph | nx.MultiGraph :
        
        if entity_attributes is None:
            entity_attributes = []

        if directed and multi_edges:
            G = nx.MultiDiGraph()
        elif directed and not multi_edges:
            G = nx.DiGraph()
        elif not directed and multi_edges:
            G = nx.MultiGraph()
        else:
            G = nx.Graph

        filtered_entities = entities
        if len(allowed_entity_types) > 0:
            filtered_entities = [entity for entity in entities if entity[self.ENTITY_TYPE] in allowed_entity_types]
        
        entities_as_dict = {entity[self.ENTITY_ID] : entity for entity in filtered_entities}
        for entity in filtered_entities:
            attributes = {}
            for attribute in entity_attributes:
                if entity.get(attribute, None) is not None:
                    attributes[attribute] = entity[attribute]
            if len(attributes) > 0:
                G.add_node(entity[self.ENTITY_ID], **attributes)
            else:
                G.add_node(entity[self.ENTITY_ID])
        
        ###### dependency projection to omit overwriting dependenccies without having defined a multigraph ######
        if multi_edges:
            for relation in relations:
                _allowed_type = True
                if len(allowed_relation_types) > 0:
                    if relation[self.ENTITY_RELATION_TYPE] not in allowed_relation_types:
                        _allowed_type = False

                if _allowed_type:
                    _add = True
                    source = relation[self.ENTITY_RELATION_FROM]
                    target = relation[self.ENTITY_RELATION_TO]
                    
                    if not self_loops: # self loops are not allowed
                        if source == target:
                            _add = False

                    if source not in entities_as_dict:
                        _add = False
                        print(f"node with entityId {source} not in data.entities, but in relations. will not be added")
                    
                    if target not in entities_as_dict:
                        _add = False
                        print(f"node with entityId {target} not in data.entities, but in relations. will not be added")

                    if _add:
                        if weight is not None:
                            attributes = {weight: 1, self.ENTITY_RELATION_TYPE: relation[self.ENTITY_RELATION_TYPE]}
                        else:
                            attributes = {self.ENTITY_RELATION_TYPE: relation[self.ENTITY_RELATION_TYPE]}
                        G.add_edge(source, target, **attributes)

        else: ####### no multi edges #######

            projection = {} #{(from, to): weight}
            for relation in relations:
                _allowed_type = True
                if len(allowed_relation_types) > 0:
                    if relation[self.ENTITY_RELATION_TYPE] not in allowed_relation_types:
                        _allowed_type = False

                if _allowed_type:
                    _add = True
                    source = relation[self.ENTITY_RELATION_FROM]
                    target = relation[self.ENTITY_RELATION_TO]
                    
                    if not self_loops: # self loops are not allowed
                        if source == target:
                            _add = False

                    if source not in entities_as_dict:
                        _add = False
                        print(f"node with entityId {source} not in data.entities, but in relations. will not be added")
                    
                    if target not in entities_as_dict:
                        _add = False
                        print(f"node with entityId {target} not in data.entities, but in relations. will not be added")

                    if _add:
                        relation_key = (source, target)                    
                        if relation_key not in projection:
                            projection[relation_key] = 0
                        projection[relation_key] += 1

            for relation_key in projection:
                (source, target) = relation_key
                if weight is not None:
                    weight_attribute = {weight: projection[relation_key]}
                    G.add_edge(source, target, **weight_attribute)
                else:
                    G.add_edge(source, target)
    
        if not isolated_nodes: #remove isolated nodes from graph
            #in directed graph degree is the sum of in_degree and out_degree
            isolated_nodes = [node for (node, degree) in G.degree() if degree == 0]
            print(f"remove {len(isolated_nodes)} isolated nodes")
            for node in isolated_nodes:
                G.remove_node(node)
        
        if clustering is not None: #remove all nodes without clustering attribute from graph
            nodes_without_clustering_attribute = [node for node in G.nodes() if len(G.nodes[node].get(clustering, "")) == 0]
            for node in nodes_without_clustering_attribute:
                G.remove_node(node)

        return G

    ############### Make the call graph entities from pdg entities ############
    def make_call_graph_entities(self, entities: List[dict], allowed_entity_types: List[str]=None) -> List[dict]: 
        if self.exxfer_model == "JAVA":
            raise RuntimeError(f"not implemented for model {self.exxfer_model }")
        
        if allowed_entity_types is not None and len(allowed_entity_types) > 0:
            filtered_entities = [entity for entity in entities if entity[self.ENTITY_TYPE] in allowed_entity_types]
        else:
            filtered_entities = entities
            
        def new_element(entityId: int, method:dict, origin_id:str, origin_fqn:str) -> dict:
            element = method 
            #### add additional attributes 
            element[self.ELEMENT_ID] = entityId
            element[self.ELEMENT_FQN] = method[self.ENTITY_METHOD_FQN]
            element[self.ELEMENT_ORIGIN_ID] = origin_id
            element[self.ELEMENT_ORIGIN_FQN] = origin_fqn
            element[self.ELEMENT_IS_EP] = method[self.ENTITY_METHOD_IS_EP]
            element[self.ELEMENT_IS_EX] = False


            return element

        entityId = -1
        elements = []
        for entity in filtered_entities:
            
            for method in entity[self.ENTITY_METHODS]:
                entityId += 1 
                element = new_element(entityId, 
                                        method, 
                                        entity[self.ENTITY_ID], 
                                        entity[self.ENTITY_FQN])
                
                elements.append(element)

        return elements

    ################ Make the call graph relations from cg_entities and #######
    def make_call_graph_relations(self, elements: List[dict]) -> List[dict]:
        if self.exxfer_model == "JAVA":
            raise RuntimeError(f"not implemented for model {self.exxfer_model }")

        elements_dict = {element[self.ELEMENT_FQN]: element for element in elements}
        relationId = -1
        relations = []
        for element in elements:
            for invocation in element[self.ENTITY_METHOD_INVOCATIONS]:
                if invocation in elements_dict:
                    source = element[self.ELEMENT_ID]
                    target = elements_dict[invocation][self.ELEMENT_ID]
                    
                    relation = {}
                    relationId += 1
                    relation[self.ELEMENT_RELATION_ID] = relationId
                    relation[self.ELEMENT_RELATION_FROM] = source
                    relation[self.ELEMENT_RELATION_TO] = target
                    relation[self.ELEMENT_RELATION_TYPE] = self.ELEMENT_RELATION_TYPE_METHOD_TO_METHOD
                    
                    relations.append(relation)
                else:
                    print(f"there is no method identity for method invocation {invocation} in method {element[self.ENTITY_METHOD_INVOCATIONS]}")

        return relations

    def enrich_exitpoint(self, 
                        elements: List[dict], 
                        entities: List[dict], 
                        entity_relations: List[dict]):
        ex_counter = 0

        entities_dict = {entity['entityId']: entity for entity in entities }

        relations_by_type = self.get_entity_relations_by_type(relations=entity_relations, allowed_entities=entities)

        potential_exitpoints = set()
        for relation in relations_by_type.get(self.ENTITY_RELATION_TYPE_ORM,[]):
            source = entities_dict.get(relation[self.ENTITY_RELATION_FROM], None) 
            target = entities_dict.get(relation[self.ENTITY_RELATION_TO], None)
            
            if source and target:
                potential_exitpoints.add(relation[self.ENTITY_RELATION_FROM])

        for relation in relations_by_type.get(self.ENTITY_RELATION_TYPE_CRUD,[]):
            source = entities_dict.get(relation[self.ENTITY_RELATION_FROM],None)
            target = entities_dict.get(relation[self.ENTITY_RELATION_TO], None)
            if source and target:
                potential_exitpoints.add(relation[self.ENTITY_RELATION_FROM])

        for element in elements:
            origin_id = element[self.ELEMENT_ORIGIN_ID]
            if origin_id in potential_exitpoints:
                element[self.ELEMENT_IS_EX] = True
                ex_counter += 1

        return ex_counter

    ###############  Build the call graph ####################################
    def build_call_graph(self, 
                        elements: List[dict],
                        element_attributes=[],
                        element_relations: List[dict]=None,
                        directed: bool=True, 
                        isolated_nodes=True,
                        weight: str=None,
                        selfloop=False) -> nx.Graph | nx.DiGraph:
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
    
        #if self.exxfer_model == EXXFER_MODEL_PYTHON_2:
        #    element_attributes.append(self.ENTITY_METHOD_PARAMETERS)

        for element in elements:
            attributes = {}
            for attribute in element_attributes:
                if element.get(attribute, None) is not None:
                    attributes[attribute] = element[attribute]
            if len(attributes) > 0:
                G.add_node(element[self.ELEMENT_ID], **attributes)
            else:
                G.add_node(element[self.ELEMENT_ID])

        if element_relations is None:    
            relations = self.make_call_graph_relations(elements=elements)
        else:
            relations = element_relations

        edges_projection = {}

        for relation in relations:
            _add = True
            _reason = ""
            source = relation[self.ELEMENT_RELATION_FROM] 
            if source not in G.nodes:
                _add = False
                _reason = "source not in graph"
            target = relation[self.ELEMENT_RELATION_TO]
            if target not in G.nodes:
                _add = False
                _reason = "target not in graph"
            if not selfloop:
                if source == target:
                    _add = False
                    _reason = "selfloop"
            
            if _add:
                if (source, target) not in edges_projection:
                    edges_projection[(source, target)] = 0
                edges_projection[(source, target)] += 1
            else:
                #print(f"invocation {relation} could not been added as an edge, reason: {_reason}")
                pass

        if weight is not None:
            edges = [(source, target, {weight: edges_projection[(source, target)]}) for (source, target) in edges_projection]
        
        else:
            edges = [(source, target, {weight: 1}) for (source, target) in edges_projection]
        
        G.add_edges_from(edges)


        if not isolated_nodes: #remove isolated nodes from graph
            
            #directed graph degree is the sum of in_degree and out_degree
            isolated_nodes = [node for (node, degree) in G.degree() if degree == 0]
            print(f"remove {len(isolated_nodes)} isolated nodes")
            for node in isolated_nodes:
                G.remove_node(node)

        return G

    ################ Build BiPartite Graph from cg_entities and pdg_entities #
    def build_element_to_entity_graph(  self,
                                        entity_graph: nx.DiGraph,
                                        element_graph: nx.DiGraph,
                                        ignore_touched_classes_inside_elements:bool = False):
        """ build a bipartite graph between call graph elements and program graph entities
        
            to distinguish the two node set we use as identifier for call graph elements
            and for program graph entities their fullQualifiedName.

            The edges are formed by all relations between these two types of nodes:

            - an element as a method is allways part of an entity (its origin)
            - an element as method could have parameteres in their signature. Parameters might 
              be the source of relations to other entities
            - an elenemt as a method could have fields inside method body. 
              Fields might be the source of relations to other entities 
        
        entity_graph:
        ------------------------- entity keys    --------------------------------------
        ['id', 'comments', 'endLine', 'entityFullQualifiedName', 'entityId', 'entityName', 
        'entityOrigin', 'entityType', 'entrypoint', 'exitpoint', 'javaDoc', 'language', 
        'loc', 'package', 'qloc', 'startLine', 'visibility']
        ------------------------- relation keys  --------------------------------------
        ['source', 'target', 'from', 'methodFullQualifiedName', 'relationId', 'relationType', 'to']
        element_graph:
        ------------------------- element keys    --------------------------------------
        ['id', 'elementFullQualifiedName', 'elementId', 'elementName', 'elementType', 'entrypoint', 
        'exitpoint', 'parentEntityFullQualifiedName', 'parentEntityOrigin']
        ------------------------- relation keys  --------------------------------------
        ['source', 'target', 'from', 'invocationId', 'invocationType', 'to']
        
        """
        entities_fqn = [entity_graph.nodes[node_id][self.ENTITY_FQN] for node_id in entity_graph.nodes()]


        B = nx.Graph()
        for element_id in element_graph.nodes():
            element = element_graph.nodes[element_id]
            element_fqn = element[self.ELEMENT_FQN]
            B.add_node(element_fqn, bipartite=0)
            origin_id = element.get(self.ELEMENT_ORIGIN_ID, None) #element[self.ELEMENT_ORIGIN_ID]
            
            if entity_graph.has_node(origin_id):
                origin_fqn = element.get(self.ELEMENT_ORIGIN_FQN, None) #element[self.ELEMENT_ORIGIN_FQN]
                B.add_node(origin_fqn, bipartite=1)
                B.add_edge(element_fqn, origin_fqn)

                if not ignore_touched_classes_inside_elements:
                    
                    if self.exxfer_model == EXXFER_MODEL_PYTHON_2:                        
                        element_parameters = [param[self.ENTITY_METHOD_PARAMETER_TYPE] for param in element[self.ENTITY_METHOD_PARAMETERS]]
                        element_fields = [field for field in element[self.ENTITY_METHOD_FIELDS]]

                        element_parameter_and_fields = set(element_parameters).union(set(element_fields))


                    elif self.exxfer_model == EXXFER_MODEL_JAVA:

                        element_parameter_and_fields = set()                        
                        for successor_id in entity_graph.successors(origin_id):
                            edge = entity_graph.get_edge_data(origin_id, successor_id)
                            if edge is not None:
                                method_fqn = edge.get(self.ENTITY_RELATION_METHOD, None)
                                if method_fqn == element_fqn:
                                    successor = entity_graph.nodes(successor_id)
                                    element_parameter_and_fields.add(successor[self.ENTITY_FQN])

                    else:
                        element_parameter_and_fields = set()

                    

                    for entity_fqn in  element_parameter_and_fields:
                        if entity_fqn in entities_fqn:
                            B.add_node(entity_fqn, bipartite=1)
                            B.add_edge(element_fqn, entity_fqn)


        return B

    def get_touched_entities(self, 
                            G: nx.MultiDiGraph, 
                            CG: nx.DiGraph, 
                            strategy="METHOD_ENTRYPOINT") -> Set[str]:
        """ Retrieves touched entities:
            
            Touched entities are all entities that are touched iterating on paths 

            There are three strategies for identifying toched entities:

            - Strategy 1:   ZERO_DEGREE
                            iterate over all entrypoints as nodes with in_degree==0. 
                            And follow all paths starting with an entrypoint. 

            - Strategy 2:   ENTRYPOINT
                            iterate over all element entrypoints "entrypoint" == True.
                            And follow all paths starting with an entrypoint.

            - Strategy 3:   ENTRYPOINT_AND_EXITPOINT
                            iterate over all paths between entrypoints and endpoints.
            
            - Strategy 4:   ENTRYPOINT_OR_EXITPOINT
                            uses strategy 2 and in addition iterates over all paths
                            between zero degree entry points(strategy 1) and exitpoints

        """


        entity_id_to_fqn = {entity_id: G.nodes[entity_id][self.ENTITY_FQN] for entity_id in G.nodes()}
        inheritance_dict = {entity_id: [] for entity_id in G.nodes()}
        implements_dict = {entity_id: [] for entity_id in G.nodes()}
        
        x=0
        y=0
        for (source_id, target_id, k, data) in G.edges(data=True, keys=True):
            
            if data[self.ENTITY_RELATION_TYPE] == self.ENTITY_RELATION_TYPE_INHERITANCE:
                inheritance_dict[source_id].append(entity_id_to_fqn[target_id])
                x += 1
            if data[self.ENTITY_RELATION_TYPE] == self.ENTITY_RELATION_TYPE_INTERFACE:
                implements_dict[source_id].append(entity_id_to_fqn[target_id])
                y += 1
        #print(f"There are {x} INHERITANCE and {y} INTERFACE relations.")

        element_id_to_fqn = {node:CG.nodes[node][self.ELEMENT_FQN] for node in CG.nodes()}

        B = self.build_element_to_entity_graph(entity_graph=G, element_graph=CG)
        
        touched_entities = set()

        entrypoints = [] # node ids that are entrypoints

        match strategy:

            case "ENTRYPOINT":
                entrypoints = [node_id for node_id in CG.nodes() if CG.nodes[node_id].get(self.ELEMENT_IS_EP, False)]
                for ep_id in entrypoints:
                    tes = self.get_touched_entities_for_ep(source_id=ep_id, CG=CG, nodeid_to_fqn=element_id_to_fqn, 
                                                            min_trace_len=0, B=B, 
                                                            inheritance=inheritance_dict, implements=implements_dict)
                    touched_entities = touched_entities.union(tes)

            case "ZERO_DEGREE":
                entrypoints = [node_id for (node_id, degree) in CG.in_degree() if degree == 0]
                for ep_id in entrypoints:
                    tes = self.get_touched_entities_for_ep(source_id=ep_id, CG=CG, nodeid_to_fqn=element_id_to_fqn, 
                                                            min_trace_len=0, B=B,
                                                            inheritance=inheritance_dict, implements=implements_dict)
                    touched_entities = touched_entities.union(tes)

            case "ENTRYPOINT_AND_EXITPOINT":
                entrypoints = [node_id for node_id in CG.nodes() if CG.nodes[node_id].get(self.ELEMENT_IS_EP, False)]
                exitpoints = [node_id for node_id in CG.nodes() if CG.nodes[node_id].get(self.ELEMENT_IS_EX, False)]
                for ep_id in entrypoints:
                    for ex_id in exitpoints:
                        tes = self.get_touched_entities_for_ep_ex(source_id=ep_id, target_id=ex_id, CG=CG, 
                                                                    nodeid_to_fqn=element_id_to_fqn, min_trace_len=0, B=B,
                                                                    inheritance=inheritance_dict,
                                                                    implements=implements_dict)
                        touched_entities = touched_entities.union(tes)
                print("entrypoints:", len(entrypoints))
                print("exitpoints :", len(exitpoints))
            
            case "ENTRYPOINT_OR_EXITPOINT":
                entrypoints = [node_id for node_id in CG.nodes() if CG.nodes[node_id].get(self.ELEMENT_IS_EP, False)]
                for ep_id in entrypoints:
                    tes = self.get_touched_entities_for_ep(source_id=ep_id, CG=CG, nodeid_to_fqn=element_id_to_fqn, 
                                                            min_trace_len=0, B=B, 
                                                            inheritance=inheritance_dict, implements=implements_dict)
                    touched_entities = touched_entities.union(tes)
                zero_degree_entrypoints = [node_id for (node_id, degree) in CG.in_degree() if degree == 0]
                exitpoints = [node_id for node_id in CG.nodes() if CG.nodes[node_id].get(self.ELEMENT_IS_EX, False)]
                for ep_id in zero_degree_entrypoints:
                    for ex_id in exitpoints:
                        tes = self.get_touched_entities_for_ep_ex(source_id=ep_id, target_id=ex_id, CG=CG, 
                                                                    nodeid_to_fqn=element_id_to_fqn, min_trace_len=0, B=B,
                                                                    inheritance=inheritance_dict,
                                                                    implements=implements_dict)
                        touched_entities = touched_entities.union(tes)
                print("entrypoints            :", len(entrypoints))
                print("zero_degree_entrypoints:", len(zero_degree_entrypoints))
                print("exitpoints             :", len(exitpoints))

        return touched_entities

    def get_touched_entities_for_ep(self,
                                    source_id:str, 
                                    CG:nx.DiGraph,
                                    nodeid_to_fqn: dict,
                                    min_trace_len=0,
                                    B:nx.Graph=None, 
                                    inheritance=None, 
                                    implements=None) -> Set[str]:

        execution_trace = nx.dfs_successors(CG, source=source_id, depth_limit=None)

        executed_nodes = set()
        if len(execution_trace) >= min_trace_len:
        
            #if len(execution_trace) == 0: #There are no invocations in the method
            #    execution_trace = {source_id:[]}
            for node_id in execution_trace:
                executed_nodes.add(node_id)
                for id in execution_trace[node_id]:
                    executed_nodes.add(id)
        
        
        touched_entities = self.get_touched_entities_for_trace_(trace=list(executed_nodes), 
                                                                CG=CG, nodeid_to_fqn=nodeid_to_fqn, 
                                                                B=B,
                                                                inheritance=inheritance, 
                                                                implements=implements)
        return touched_entities

    def get_touched_entities_for_ep_ex(self, 
                                        source_id:str, target_id:str, 
                                        CG:nx.DiGraph, nodeid_to_fqn: dict, min_trace_len=0, B:nx.Graph=None, 
                                        inheritance=None, implements=None) -> List[str]:
        


        # first find all paths from entrypoint to exitpoint        
        all_path_nodes_per_ep = set()
        paths = nx.all_simple_paths(CG, source=source_id, target=target_id) # get all paths between source and target
        for path in map(nx.utils.pairwise, paths):
            #collect all elements on these paths
            for (s, t) in list(path):
                all_path_nodes_per_ep.add(s)
        

        # iterate over all found elements
        touched_entities = self.get_touched_entities_for_trace_(trace=list(all_path_nodes_per_ep), 
                                                                CG=CG, nodeid_to_fqn=nodeid_to_fqn, 
                                                                B=B,
                                                                inheritance=inheritance, 
                                                                implements=implements)


        return touched_entities

    def get_touched_entities_for_trace_(self, 
                                        trace: List[str], 
                                        CG:nx.DiGraph, nodeid_to_fqn: dict, 
                                        B: nx.DiGraph=None,
                                        inheritance=None, implements=None) -> Set[str]:
        
        touched_entities = set()
        for node_id in trace:
            # get class for each touched node
            touched_entities.add(CG.nodes[node_id][self.ELEMENT_ORIGIN_FQN])
            #  all other dependencies for touched node from BiGraph B:
            if B is not None:
                executed_node_fqn = nodeid_to_fqn[node_id]
                for fq_class_name in list(B.neighbors(executed_node_fqn)):
                    touched_entities.add(fq_class_name)
                    
                if inheritance is not None:
                    origin_id = CG.nodes[node_id][self.ELEMENT_ORIGIN_ID]
                    for entity_fqn in inheritance[origin_id]:
                        touched_entities.add(entity_fqn)
                if implements is not None:
                    origin_id = CG.nodes[node_id][self.ELEMENT_ORIGIN_ID]
                    for entity_fqn in implements[origin_id]:
                        touched_entities.add(entity_fqn)

        return touched_entities

    ############################################################################+
    
    def filter_entities_by_origin(self, entities: List[dict], origins:List[str]) -> dict:
        '''returns a list of entities belonging to origins. 
        Here origins are catagories like LIB, SRC, SRCGEN, ...'''
        filtered_entities = []
        for entity in entities:
            if entity[self.ENTITY_ORIGIN] in origins:
                filtered_entities.append(entity)
        return filtered_entities

    def filter_entity_relations_by_allowed_entities(self, relations: List[dict], filtered_entities: List[dict]) -> dict:
        '''returns a list of relations filtered by allowed entities'''
        filtered_entities_ids = [entity[self.ENTITY_ID] for entity in filtered_entities]
        filtered_relations = []
        for relation in relations:
            if relation[self.ENTITY_RELATION_FROM] in filtered_entities_ids and relation[self.ENTITY_RELATION_TO] in filtered_entities_ids:
                filtered_relations.append(relation) 
        return filtered_relations

    def filter_elements_by_origin(self, elements: List[dict], origins:List[str]) -> dict:
        '''return a list of all elements belonging to origins. 
        Here origins are catagories like LIB, SRC, SRCGEN, ...'''
        filtered_elements = []
        for entity in elements:
            if entity[self.ELEMENT_ORIGIN_ID] in origins:
                filtered_elements.append(entity)
        return filtered_elements  

    def filter_entity_relations_by_allowed_elements(self, element_relations: List[dict], filtered_elements: List[dict]) -> dict:
        '''returns a list of relations filtered by allowed elements'''
        filtered_element_ids = [e[self.ELEMENT_ID] for e in filtered_elements]
        filtered_element_relations = []
        for relation in element_relations:
            if relation[self.ELEMENT_RELATION_FROM] in filtered_element_ids and relation[self.ELEMENT_RELATION_TO] in filtered_element_ids:
                filtered_element_relations.append(relation) 
        return filtered_element_relations