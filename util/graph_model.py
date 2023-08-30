###############################################################################
#
# Exxfer Graph Model
#
# Copyright (C) 2023 EXXFER Research Project
# Author: Dominik Neumann <dominik.neumann@exxeta.com>
#
# URL: <https://exxfer.io/>
###############################################################################


EXXFER_MODEL_JAVA = "JAVA"
EXXFER_MODEL_PYTHON_2 = "PYTHON_2"

class Exxfer():
    
    ELEMENT_FQN = "elementFullQualifiedName"
    ELEMENT_ID = 'elementId'
    ELEMENT_TYPE = 'elementType'
    ELEMENT_IS_EP = "entrypoint"
    ELEMENT_IS_EX = "exitpoint"   
    ELEMENT_ORIGIN = 'parentEntityOrigin'         
    ELEMENT_ORIGIN_ID = 'parentEntityId' # only valid in python_2 exfer data model
    ELEMENT_ORIGIN_FQN = 'parentEntityFullQualifiedName'
        
    ELEMENT_RELATION_TYPE = 'invocationType'
    ELEMENT_RELATION_FROM = 'from'
    ELEMENT_RELATION_TO = 'to'
    ELEMENT_RELATION_ID = 'invocationId'

    ELEMENT_RELATION_TYPE_METHOD_TO_METHOD = 'METHOD_TO_METHOD'
    ELEMENT_RELATION_TYPE_FIELD_TO_METHOD = 'FIELD_TO_METHOD'
    ELEMENT_RELATION_TYPE_INITIALIZER_TO_METHOD = 'INITIALIZER_TO_METHOD'

    ELEMENT_TYPE_METHOD = "METHOD"
    ELEMENT_TYPE_FIELD = "FIELD" 
    ELEMENT_TYPE_INITIALIZER = "INITIALIZER" 

    ENTITY_TYPE = 'entityType'
    ENTITY_ID = 'entityId'
    ENTITY_PACKAGE = 'package'
    ENTITY_FQN = 'entityFullQualifiedName'
    ENTITY_METHODS = 'methods'
    ENTITY_ORIGIN = "entityOrigin"

    ENTITY_RELATION_TYPE_METHODPARAMETER = "METHODPARAMETER"
    ENTITY_RELATION_TYPE_METHODFIELD = "METHODFIELD"
    ENTITY_RELATION_TYPE_METHODRETURN = "METHODRETURN"
    ENTITY_RELATION_TYPE_METHODEXCEPTION = "METHODEXCEPTION"
    ENTITY_RELATION_TYPE_METHODUSEDTYPE = "METHODUSEDTYPE"
    ENTITY_RELATION_TYPE_INVOCATIONRETURN = "INVOCATIONRETURN"
    ENTITY_RELATION_TYPE_METHODVARIABLE = "METHODVARIABLE"
    ENTITY_RELATION_TYPE_ORM = "ORM"
    ENTITY_RELATION_TYPE_CRUD = "CRUD"
    ENTITY_RELATION_TYPE_INHERITANCE = "INHERITANCE"
    ENTITY_RELATION_TYPE_INTERFACE = "INTERFACE"
    ENTITY_RELATION_TYPE_ANNOTATION = "ANNOTATION"
    ENTITY_RELATION_TYPE_ANNOTATION_VALUE = "ANNOTATION_VALUE"
    ENTITY_RELATION_TYPE_FIELD = "FIELD"
    ENTITY_RELATION_TYPE_CLASSLITERAL = "CLASSLITERAL"

    ENTITY_RELATION_TYPE = 'relationType'
    ENTITY_RELATION_FROM = 'from'
    ENTITY_RELATION_TO = 'to'
    ENTITY_RELATION_METHOD = 'methodFullQualifiedName'
    ENTITY_RELATION_ID = 'relationId'

    ENTITY_RELATION_TYPE_CALL = "INVOCATION"
    ENTITY_RELATION_TYPE_INNERCLASS = "INNER"

    ENTITY_METHOD_FQN = "identity"
    ENTITY_METHOD_PARAMETERS = 'parameters'
    ENTITY_METHOD_PARAMETER_TYPE = 'type'
    ENTITY_METHOD_FIELDS = 'fields'
    ENTITY_METHOD_INVOCATIONS = 'invocations'
    ENTITY_METHOD_IS_EP = "is_entrypoint"

    ENTITY_COMMITS = "commits"
    ENTITY_COMMIT_HASH = "hash"
    ENTITY_COMMIT_DATE = "date"
    ENTITY_COMMIT_COMMITTER = "committer"
    ENTITY_COMMIT_AUTHOR = "author"


class ExxferPython2(Exxfer):

    ENTITY_NAME = 'class_name'
    ENTITY_IS_EP = "is_entrypoint"
    ENTITY_IS_EX = "is_exitpoint"

class ExxferJava(Exxfer):

    ENTITY_NAME = 'entityName'
    ENTITY_IS_EP = "entrypoint"
    ENTITY_IS_EX = "exitpoint"

