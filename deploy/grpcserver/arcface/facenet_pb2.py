# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: facenet.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='facenet.proto',
  package='deploy.grpcserver.facenet',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\rfacenet.proto\x12\x19\x64\x65ploy.grpcserver.facenet\"*\n\x0cImageMessage\x12\r\n\x05image\x18\x01 \x01(\x0c\x12\x0b\n\x03\x64im\x18\x02 \x03(\x05\"2\n\x10\x45mbeddingMessage\x12\x11\n\tembedding\x18\x01 \x01(\x0c\x12\x0b\n\x03\x64im\x18\x02 \x03(\x05\x32m\n\x0cGetEmbedding\x12]\n\x03Get\x12\'.deploy.grpcserver.facenet.ImageMessage\x1a+.deploy.grpcserver.facenet.EmbeddingMessage\"\x00\x62\x06proto3')
)




_IMAGEMESSAGE = _descriptor.Descriptor(
  name='ImageMessage',
  full_name='deploy.grpcserver.facenet.ImageMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='image', full_name='deploy.grpcserver.facenet.ImageMessage.image', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dim', full_name='deploy.grpcserver.facenet.ImageMessage.dim', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=44,
  serialized_end=86,
)


_EMBEDDINGMESSAGE = _descriptor.Descriptor(
  name='EmbeddingMessage',
  full_name='deploy.grpcserver.facenet.EmbeddingMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='embedding', full_name='deploy.grpcserver.facenet.EmbeddingMessage.embedding', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dim', full_name='deploy.grpcserver.facenet.EmbeddingMessage.dim', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=88,
  serialized_end=138,
)

DESCRIPTOR.message_types_by_name['ImageMessage'] = _IMAGEMESSAGE
DESCRIPTOR.message_types_by_name['EmbeddingMessage'] = _EMBEDDINGMESSAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ImageMessage = _reflection.GeneratedProtocolMessageType('ImageMessage', (_message.Message,), dict(
  DESCRIPTOR = _IMAGEMESSAGE,
  __module__ = 'facenet_pb2'
  # @@protoc_insertion_point(class_scope:deploy.grpcserver.facenet.ImageMessage)
  ))
_sym_db.RegisterMessage(ImageMessage)

EmbeddingMessage = _reflection.GeneratedProtocolMessageType('EmbeddingMessage', (_message.Message,), dict(
  DESCRIPTOR = _EMBEDDINGMESSAGE,
  __module__ = 'facenet_pb2'
  # @@protoc_insertion_point(class_scope:deploy.grpcserver.facenet.EmbeddingMessage)
  ))
_sym_db.RegisterMessage(EmbeddingMessage)



_GETEMBEDDING = _descriptor.ServiceDescriptor(
  name='GetEmbedding',
  full_name='deploy.grpcserver.facenet.GetEmbedding',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=140,
  serialized_end=249,
  methods=[
  _descriptor.MethodDescriptor(
    name='Get',
    full_name='deploy.grpcserver.facenet.GetEmbedding.Get',
    index=0,
    containing_service=None,
    input_type=_IMAGEMESSAGE,
    output_type=_EMBEDDINGMESSAGE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_GETEMBEDDING)

DESCRIPTOR.services_by_name['GetEmbedding'] = _GETEMBEDDING

# @@protoc_insertion_point(module_scope)