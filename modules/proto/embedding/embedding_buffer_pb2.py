# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: embedding_buffer.proto
# Protobuf Python Version: 5.28.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    1,
    '',
    'embedding_buffer.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16\x65mbedding_buffer.proto\x12\tembedding\"%\n\x10\x45mbeddingRequest\x12\x11\n\tfile_path\x18\x01 \x01(\t\"(\n\x11\x45mbeddingResponse\x12\x13\n\x0bjson_stream\x18\x01 \x01(\t2b\n\x10\x45mbeddingService\x12N\n\x0fStreamEmbedding\x12\x1b.embedding.EmbeddingRequest\x1a\x1c.embedding.EmbeddingResponse0\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'embedding_buffer_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_EMBEDDINGREQUEST']._serialized_start=37
  _globals['_EMBEDDINGREQUEST']._serialized_end=74
  _globals['_EMBEDDINGRESPONSE']._serialized_start=76
  _globals['_EMBEDDINGRESPONSE']._serialized_end=116
  _globals['_EMBEDDINGSERVICE']._serialized_start=118
  _globals['_EMBEDDINGSERVICE']._serialized_end=216
# @@protoc_insertion_point(module_scope)
