У
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718��
�
4score_convolutional_neural_network_1/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64score_convolutional_neural_network_1/conv2d_1/kernel
�
Hscore_convolutional_neural_network_1/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp4score_convolutional_neural_network_1/conv2d_1/kernel*'
_output_shapes
:�*
dtype0
�
2score_convolutional_neural_network_1/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*C
shared_name42score_convolutional_neural_network_1/conv2d_1/bias
�
Fscore_convolutional_neural_network_1/conv2d_1/bias/Read/ReadVariableOpReadVariableOp2score_convolutional_neural_network_1/conv2d_1/bias*
_output_shapes	
:�*
dtype0
�
3score_convolutional_neural_network_1/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*D
shared_name53score_convolutional_neural_network_1/dense_3/kernel
�
Gscore_convolutional_neural_network_1/dense_3/kernel/Read/ReadVariableOpReadVariableOp3score_convolutional_neural_network_1/dense_3/kernel*
_output_shapes
:	�d*
dtype0
�
1score_convolutional_neural_network_1/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*B
shared_name31score_convolutional_neural_network_1/dense_3/bias
�
Escore_convolutional_neural_network_1/dense_3/bias/Read/ReadVariableOpReadVariableOp1score_convolutional_neural_network_1/dense_3/bias*
_output_shapes
:d*
dtype0
�
3score_convolutional_neural_network_1/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*D
shared_name53score_convolutional_neural_network_1/dense_4/kernel
�
Gscore_convolutional_neural_network_1/dense_4/kernel/Read/ReadVariableOpReadVariableOp3score_convolutional_neural_network_1/dense_4/kernel*
_output_shapes

:dd*
dtype0
�
1score_convolutional_neural_network_1/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*B
shared_name31score_convolutional_neural_network_1/dense_4/bias
�
Escore_convolutional_neural_network_1/dense_4/bias/Read/ReadVariableOpReadVariableOp1score_convolutional_neural_network_1/dense_4/bias*
_output_shapes
:d*
dtype0
�
3score_convolutional_neural_network_1/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*D
shared_name53score_convolutional_neural_network_1/dense_5/kernel
�
Gscore_convolutional_neural_network_1/dense_5/kernel/Read/ReadVariableOpReadVariableOp3score_convolutional_neural_network_1/dense_5/kernel*
_output_shapes

:d*
dtype0
�
1score_convolutional_neural_network_1/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31score_convolutional_neural_network_1/dense_5/bias
�
Escore_convolutional_neural_network_1/dense_5/bias/Read/ReadVariableOpReadVariableOp1score_convolutional_neural_network_1/dense_5/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
conv
flatten
hidden1
hidden2
outlayer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
h

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
8
0
1
2
3
4
5
!6
"7
 
8
0
1
2
3
4
5
!6
"7
�
	variables
regularization_losses
'layer_metrics
(layer_regularization_losses
)metrics
*non_trainable_variables
trainable_variables

+layers
 
pn
VARIABLE_VALUE4score_convolutional_neural_network_1/conv2d_1/kernel&conv/kernel/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE2score_convolutional_neural_network_1/conv2d_1/bias$conv/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
	variables
regularization_losses
,layer_metrics
-layer_regularization_losses
.metrics
/non_trainable_variables
trainable_variables

0layers
 
 
 
�
	variables
regularization_losses
1layer_metrics
2layer_regularization_losses
3metrics
4non_trainable_variables
trainable_variables

5layers
rp
VARIABLE_VALUE3score_convolutional_neural_network_1/dense_3/kernel)hidden1/kernel/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1score_convolutional_neural_network_1/dense_3/bias'hidden1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
	variables
regularization_losses
6layer_metrics
7layer_regularization_losses
8metrics
9non_trainable_variables
trainable_variables

:layers
rp
VARIABLE_VALUE3score_convolutional_neural_network_1/dense_4/kernel)hidden2/kernel/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE1score_convolutional_neural_network_1/dense_4/bias'hidden2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
	variables
regularization_losses
;layer_metrics
<layer_regularization_losses
=metrics
>non_trainable_variables
trainable_variables

?layers
sq
VARIABLE_VALUE3score_convolutional_neural_network_1/dense_5/kernel*outlayer/kernel/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE1score_convolutional_neural_network_1/dense_5/bias(outlayer/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
�
#	variables
$regularization_losses
@layer_metrics
Alayer_regularization_losses
Bmetrics
Cnon_trainable_variables
%trainable_variables

Dlayers
 
 
 
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
serving_default_input_1Placeholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_14score_convolutional_neural_network_1/conv2d_1/kernel2score_convolutional_neural_network_1/conv2d_1/bias3score_convolutional_neural_network_1/dense_3/kernel1score_convolutional_neural_network_1/dense_3/bias3score_convolutional_neural_network_1/dense_4/kernel1score_convolutional_neural_network_1/dense_4/bias3score_convolutional_neural_network_1/dense_5/kernel1score_convolutional_neural_network_1/dense_5/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_51522293
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameHscore_convolutional_neural_network_1/conv2d_1/kernel/Read/ReadVariableOpFscore_convolutional_neural_network_1/conv2d_1/bias/Read/ReadVariableOpGscore_convolutional_neural_network_1/dense_3/kernel/Read/ReadVariableOpEscore_convolutional_neural_network_1/dense_3/bias/Read/ReadVariableOpGscore_convolutional_neural_network_1/dense_4/kernel/Read/ReadVariableOpEscore_convolutional_neural_network_1/dense_4/bias/Read/ReadVariableOpGscore_convolutional_neural_network_1/dense_5/kernel/Read/ReadVariableOpEscore_convolutional_neural_network_1/dense_5/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_51522430
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename4score_convolutional_neural_network_1/conv2d_1/kernel2score_convolutional_neural_network_1/conv2d_1/bias3score_convolutional_neural_network_1/dense_3/kernel1score_convolutional_neural_network_1/dense_3/bias3score_convolutional_neural_network_1/dense_4/kernel1score_convolutional_neural_network_1/dense_4/bias3score_convolutional_neural_network_1/dense_5/kernel1score_convolutional_neural_network_1/dense_5/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_51522464��
�	
�
&__inference_signature_wrapper_51522293
input_1"
unknown:�
	unknown_0:	�
	unknown_1:	�d
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_515221192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
H
,__inference_flatten_1_layer_call_fn_51522324

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_515221492
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_3_layer_call_fn_51522344

inputs
unknown:	�d
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_515221622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_4_layer_call_and_return_conditional_losses_51522355

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
*__inference_dense_4_layer_call_fn_51522364

inputs
unknown:dd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_515221792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�"
�
!__inference__traced_save_51522430
file_prefixS
Osavev2_score_convolutional_neural_network_1_conv2d_1_kernel_read_readvariableopQ
Msavev2_score_convolutional_neural_network_1_conv2d_1_bias_read_readvariableopR
Nsavev2_score_convolutional_neural_network_1_dense_3_kernel_read_readvariableopP
Lsavev2_score_convolutional_neural_network_1_dense_3_bias_read_readvariableopR
Nsavev2_score_convolutional_neural_network_1_dense_4_kernel_read_readvariableopP
Lsavev2_score_convolutional_neural_network_1_dense_4_bias_read_readvariableopR
Nsavev2_score_convolutional_neural_network_1_dense_5_kernel_read_readvariableopP
Lsavev2_score_convolutional_neural_network_1_dense_5_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B&conv/kernel/.ATTRIBUTES/VARIABLE_VALUEB$conv/bias/.ATTRIBUTES/VARIABLE_VALUEB)hidden1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'hidden1/bias/.ATTRIBUTES/VARIABLE_VALUEB)hidden2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'hidden2/bias/.ATTRIBUTES/VARIABLE_VALUEB*outlayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB(outlayer/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Osavev2_score_convolutional_neural_network_1_conv2d_1_kernel_read_readvariableopMsavev2_score_convolutional_neural_network_1_conv2d_1_bias_read_readvariableopNsavev2_score_convolutional_neural_network_1_dense_3_kernel_read_readvariableopLsavev2_score_convolutional_neural_network_1_dense_3_bias_read_readvariableopNsavev2_score_convolutional_neural_network_1_dense_4_kernel_read_readvariableopLsavev2_score_convolutional_neural_network_1_dense_4_bias_read_readvariableopNsavev2_score_convolutional_neural_network_1_dense_5_kernel_read_readvariableopLsavev2_score_convolutional_neural_network_1_dense_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*b
_input_shapesQ
O: :�:�:	�d:d:dd:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�d: 

_output_shapes
:d:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::	

_output_shapes
: 
�	
�
E__inference_dense_5_layer_call_and_return_conditional_losses_51522374

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
�
G__inference_score_convolutional_neural_network_1_layer_call_fn_51522224
input_1"
unknown:�
	unknown_0:	�
	unknown_1:	�d
	unknown_2:d
	unknown_3:dd
	unknown_4:d
	unknown_5:d
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *k
ffRd
b__inference_score_convolutional_neural_network_1_layer_call_and_return_conditional_losses_515222022
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_51522149

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_1_layer_call_and_return_conditional_losses_51522304

inputs9
conv2d_readvariableop_resource:�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
E__inference_dense_5_layer_call_and_return_conditional_losses_51522195

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
b__inference_score_convolutional_neural_network_1_layer_call_and_return_conditional_losses_51522202
input_1,
conv2d_1_51522138:� 
conv2d_1_51522140:	�#
dense_3_51522163:	�d
dense_3_51522165:d"
dense_4_51522180:dd
dense_4_51522182:d"
dense_5_51522196:d
dense_5_51522198:
identity�� conv2d_1/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCallb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinput_1ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2

ExpandDimsr
CastCastExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:���������2
Cast�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallCast:y:0conv2d_1_51522138conv2d_1_51522140*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_515221372"
 conv2d_1/StatefulPartitionedCall�
flatten_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_515221492
flatten_1/PartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_51522163dense_3_51522165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_515221622!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_51522180dense_4_51522182*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_515221792!
dense_4/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_51522196dense_5_51522198*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_515221952!
dense_5/StatefulPartitionedCall�
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0!^conv2d_1/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�N
�

#__inference__wrapped_model_51522119
input_1g
Lscore_convolutional_neural_network_1_conv2d_1_conv2d_readvariableop_resource:�\
Mscore_convolutional_neural_network_1_conv2d_1_biasadd_readvariableop_resource:	�^
Kscore_convolutional_neural_network_1_dense_3_matmul_readvariableop_resource:	�dZ
Lscore_convolutional_neural_network_1_dense_3_biasadd_readvariableop_resource:d]
Kscore_convolutional_neural_network_1_dense_4_matmul_readvariableop_resource:ddZ
Lscore_convolutional_neural_network_1_dense_4_biasadd_readvariableop_resource:d]
Kscore_convolutional_neural_network_1_dense_5_matmul_readvariableop_resource:dZ
Lscore_convolutional_neural_network_1_dense_5_biasadd_readvariableop_resource:
identity��Dscore_convolutional_neural_network_1/conv2d_1/BiasAdd/ReadVariableOp�Cscore_convolutional_neural_network_1/conv2d_1/Conv2D/ReadVariableOp�Cscore_convolutional_neural_network_1/dense_3/BiasAdd/ReadVariableOp�Bscore_convolutional_neural_network_1/dense_3/MatMul/ReadVariableOp�Cscore_convolutional_neural_network_1/dense_4/BiasAdd/ReadVariableOp�Bscore_convolutional_neural_network_1/dense_4/MatMul/ReadVariableOp�Cscore_convolutional_neural_network_1/dense_5/BiasAdd/ReadVariableOp�Bscore_convolutional_neural_network_1/dense_5/MatMul/ReadVariableOp�
3score_convolutional_neural_network_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :25
3score_convolutional_neural_network_1/ExpandDims/dim�
/score_convolutional_neural_network_1/ExpandDims
ExpandDimsinput_1<score_convolutional_neural_network_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������21
/score_convolutional_neural_network_1/ExpandDims�
)score_convolutional_neural_network_1/CastCast8score_convolutional_neural_network_1/ExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:���������2+
)score_convolutional_neural_network_1/Cast�
Cscore_convolutional_neural_network_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpLscore_convolutional_neural_network_1_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02E
Cscore_convolutional_neural_network_1/conv2d_1/Conv2D/ReadVariableOp�
4score_convolutional_neural_network_1/conv2d_1/Conv2DConv2D-score_convolutional_neural_network_1/Cast:y:0Kscore_convolutional_neural_network_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
26
4score_convolutional_neural_network_1/conv2d_1/Conv2D�
Dscore_convolutional_neural_network_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpMscore_convolutional_neural_network_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02F
Dscore_convolutional_neural_network_1/conv2d_1/BiasAdd/ReadVariableOp�
5score_convolutional_neural_network_1/conv2d_1/BiasAddBiasAdd=score_convolutional_neural_network_1/conv2d_1/Conv2D:output:0Lscore_convolutional_neural_network_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������27
5score_convolutional_neural_network_1/conv2d_1/BiasAdd�
2score_convolutional_neural_network_1/conv2d_1/ReluRelu>score_convolutional_neural_network_1/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:����������24
2score_convolutional_neural_network_1/conv2d_1/Relu�
4score_convolutional_neural_network_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  26
4score_convolutional_neural_network_1/flatten_1/Const�
6score_convolutional_neural_network_1/flatten_1/ReshapeReshape@score_convolutional_neural_network_1/conv2d_1/Relu:activations:0=score_convolutional_neural_network_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:����������28
6score_convolutional_neural_network_1/flatten_1/Reshape�
Bscore_convolutional_neural_network_1/dense_3/MatMul/ReadVariableOpReadVariableOpKscore_convolutional_neural_network_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02D
Bscore_convolutional_neural_network_1/dense_3/MatMul/ReadVariableOp�
3score_convolutional_neural_network_1/dense_3/MatMulMatMul?score_convolutional_neural_network_1/flatten_1/Reshape:output:0Jscore_convolutional_neural_network_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d25
3score_convolutional_neural_network_1/dense_3/MatMul�
Cscore_convolutional_neural_network_1/dense_3/BiasAdd/ReadVariableOpReadVariableOpLscore_convolutional_neural_network_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02E
Cscore_convolutional_neural_network_1/dense_3/BiasAdd/ReadVariableOp�
4score_convolutional_neural_network_1/dense_3/BiasAddBiasAdd=score_convolutional_neural_network_1/dense_3/MatMul:product:0Kscore_convolutional_neural_network_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d26
4score_convolutional_neural_network_1/dense_3/BiasAdd�
1score_convolutional_neural_network_1/dense_3/ReluRelu=score_convolutional_neural_network_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������d23
1score_convolutional_neural_network_1/dense_3/Relu�
Bscore_convolutional_neural_network_1/dense_4/MatMul/ReadVariableOpReadVariableOpKscore_convolutional_neural_network_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02D
Bscore_convolutional_neural_network_1/dense_4/MatMul/ReadVariableOp�
3score_convolutional_neural_network_1/dense_4/MatMulMatMul?score_convolutional_neural_network_1/dense_3/Relu:activations:0Jscore_convolutional_neural_network_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d25
3score_convolutional_neural_network_1/dense_4/MatMul�
Cscore_convolutional_neural_network_1/dense_4/BiasAdd/ReadVariableOpReadVariableOpLscore_convolutional_neural_network_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02E
Cscore_convolutional_neural_network_1/dense_4/BiasAdd/ReadVariableOp�
4score_convolutional_neural_network_1/dense_4/BiasAddBiasAdd=score_convolutional_neural_network_1/dense_4/MatMul:product:0Kscore_convolutional_neural_network_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d26
4score_convolutional_neural_network_1/dense_4/BiasAdd�
1score_convolutional_neural_network_1/dense_4/ReluRelu=score_convolutional_neural_network_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������d23
1score_convolutional_neural_network_1/dense_4/Relu�
Bscore_convolutional_neural_network_1/dense_5/MatMul/ReadVariableOpReadVariableOpKscore_convolutional_neural_network_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02D
Bscore_convolutional_neural_network_1/dense_5/MatMul/ReadVariableOp�
3score_convolutional_neural_network_1/dense_5/MatMulMatMul?score_convolutional_neural_network_1/dense_4/Relu:activations:0Jscore_convolutional_neural_network_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������25
3score_convolutional_neural_network_1/dense_5/MatMul�
Cscore_convolutional_neural_network_1/dense_5/BiasAdd/ReadVariableOpReadVariableOpLscore_convolutional_neural_network_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02E
Cscore_convolutional_neural_network_1/dense_5/BiasAdd/ReadVariableOp�
4score_convolutional_neural_network_1/dense_5/BiasAddBiasAdd=score_convolutional_neural_network_1/dense_5/MatMul:product:0Kscore_convolutional_neural_network_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������26
4score_convolutional_neural_network_1/dense_5/BiasAdd�
IdentityIdentity=score_convolutional_neural_network_1/dense_5/BiasAdd:output:0E^score_convolutional_neural_network_1/conv2d_1/BiasAdd/ReadVariableOpD^score_convolutional_neural_network_1/conv2d_1/Conv2D/ReadVariableOpD^score_convolutional_neural_network_1/dense_3/BiasAdd/ReadVariableOpC^score_convolutional_neural_network_1/dense_3/MatMul/ReadVariableOpD^score_convolutional_neural_network_1/dense_4/BiasAdd/ReadVariableOpC^score_convolutional_neural_network_1/dense_4/MatMul/ReadVariableOpD^score_convolutional_neural_network_1/dense_5/BiasAdd/ReadVariableOpC^score_convolutional_neural_network_1/dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2�
Dscore_convolutional_neural_network_1/conv2d_1/BiasAdd/ReadVariableOpDscore_convolutional_neural_network_1/conv2d_1/BiasAdd/ReadVariableOp2�
Cscore_convolutional_neural_network_1/conv2d_1/Conv2D/ReadVariableOpCscore_convolutional_neural_network_1/conv2d_1/Conv2D/ReadVariableOp2�
Cscore_convolutional_neural_network_1/dense_3/BiasAdd/ReadVariableOpCscore_convolutional_neural_network_1/dense_3/BiasAdd/ReadVariableOp2�
Bscore_convolutional_neural_network_1/dense_3/MatMul/ReadVariableOpBscore_convolutional_neural_network_1/dense_3/MatMul/ReadVariableOp2�
Cscore_convolutional_neural_network_1/dense_4/BiasAdd/ReadVariableOpCscore_convolutional_neural_network_1/dense_4/BiasAdd/ReadVariableOp2�
Bscore_convolutional_neural_network_1/dense_4/MatMul/ReadVariableOpBscore_convolutional_neural_network_1/dense_4/MatMul/ReadVariableOp2�
Cscore_convolutional_neural_network_1/dense_5/BiasAdd/ReadVariableOpCscore_convolutional_neural_network_1/dense_5/BiasAdd/ReadVariableOp2�
Bscore_convolutional_neural_network_1/dense_5/MatMul/ReadVariableOpBscore_convolutional_neural_network_1/dense_5/MatMul/ReadVariableOp:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_51522319

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_1_layer_call_and_return_conditional_losses_51522137

inputs9
conv2d_readvariableop_resource:�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_3_layer_call_and_return_conditional_losses_51522335

inputs1
matmul_readvariableop_resource:	�d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_5_layer_call_fn_51522383

inputs
unknown:d
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_515221952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
+__inference_conv2d_1_layer_call_fn_51522313

inputs"
unknown:�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_515221372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_4_layer_call_and_return_conditional_losses_51522179

inputs0
matmul_readvariableop_resource:dd-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�*
�
$__inference__traced_restore_51522464
file_prefix`
Eassignvariableop_score_convolutional_neural_network_1_conv2d_1_kernel:�T
Eassignvariableop_1_score_convolutional_neural_network_1_conv2d_1_bias:	�Y
Fassignvariableop_2_score_convolutional_neural_network_1_dense_3_kernel:	�dR
Dassignvariableop_3_score_convolutional_neural_network_1_dense_3_bias:dX
Fassignvariableop_4_score_convolutional_neural_network_1_dense_4_kernel:ddR
Dassignvariableop_5_score_convolutional_neural_network_1_dense_4_bias:dX
Fassignvariableop_6_score_convolutional_neural_network_1_dense_5_kernel:dR
Dassignvariableop_7_score_convolutional_neural_network_1_dense_5_bias:

identity_9��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B&conv/kernel/.ATTRIBUTES/VARIABLE_VALUEB$conv/bias/.ATTRIBUTES/VARIABLE_VALUEB)hidden1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'hidden1/bias/.ATTRIBUTES/VARIABLE_VALUEB)hidden2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'hidden2/bias/.ATTRIBUTES/VARIABLE_VALUEB*outlayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB(outlayer/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpEassignvariableop_score_convolutional_neural_network_1_conv2d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpEassignvariableop_1_score_convolutional_neural_network_1_conv2d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpFassignvariableop_2_score_convolutional_neural_network_1_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpDassignvariableop_3_score_convolutional_neural_network_1_dense_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpFassignvariableop_4_score_convolutional_neural_network_1_dense_4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpDassignvariableop_5_score_convolutional_neural_network_1_dense_4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpFassignvariableop_6_score_convolutional_neural_network_1_dense_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpDassignvariableop_7_score_convolutional_neural_network_1_dense_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8�

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
E__inference_dense_3_layer_call_and_return_conditional_losses_51522162

inputs1
matmul_readvariableop_resource:	�d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�~
�
conv
flatten
hidden1
hidden2
outlayer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
E_default_save_signature
*F&call_and_return_all_conditional_losses
G__call__"�
_tf_keras_model�{"name": "score_convolutional_neural_network_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "ScoreConvolutionalNeuralNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [420, 6, 7]}, "int8", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "ScoreConvolutionalNeuralNetwork"}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*H&call_and_return_all_conditional_losses
I__call__"�

_tf_keras_layer�
{"name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 7, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 7, 1]}, "dtype": "float32", "filters": 150, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [420, 6, 7, 1]}}
�
	variables
regularization_losses
trainable_variables
	keras_api
*J&call_and_return_all_conditional_losses
K__call__"�
_tf_keras_layer�{"name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 5}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*L&call_and_return_all_conditional_losses
M__call__"�
_tf_keras_layer�{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1800}}, "shared_object_id": 9}, "build_input_shape": {"class_name": "TensorShape", "items": [420, 1800]}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
*N&call_and_return_all_conditional_losses
O__call__"�
_tf_keras_layer�{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 13}, "build_input_shape": {"class_name": "TensorShape", "items": [420, 100]}}
�

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
*P&call_and_return_all_conditional_losses
Q__call__"�
_tf_keras_layer�{"name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [420, 100]}}
X
0
1
2
3
4
5
!6
"7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
!6
"7"
trackable_list_wrapper
�
	variables
regularization_losses
'layer_metrics
(layer_regularization_losses
)metrics
*non_trainable_variables
trainable_variables

+layers
G__call__
E_default_save_signature
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
,
Rserving_default"
signature_map
O:M�24score_convolutional_neural_network_1/conv2d_1/kernel
A:?�22score_convolutional_neural_network_1/conv2d_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
	variables
regularization_losses
,layer_metrics
-layer_regularization_losses
.metrics
/non_trainable_variables
trainable_variables

0layers
I__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
regularization_losses
1layer_metrics
2layer_regularization_losses
3metrics
4non_trainable_variables
trainable_variables

5layers
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
F:D	�d23score_convolutional_neural_network_1/dense_3/kernel
?:=d21score_convolutional_neural_network_1/dense_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
	variables
regularization_losses
6layer_metrics
7layer_regularization_losses
8metrics
9non_trainable_variables
trainable_variables

:layers
M__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
E:Cdd23score_convolutional_neural_network_1/dense_4/kernel
?:=d21score_convolutional_neural_network_1/dense_4/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
	variables
regularization_losses
;layer_metrics
<layer_regularization_losses
=metrics
>non_trainable_variables
trainable_variables

?layers
O__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
E:Cd23score_convolutional_neural_network_1/dense_5/kernel
?:=21score_convolutional_neural_network_1/dense_5/bias
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
�
#	variables
$regularization_losses
@layer_metrics
Alayer_regularization_losses
Bmetrics
Cnon_trainable_variables
%trainable_variables

Dlayers
Q__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
#__inference__wrapped_model_51522119�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"
input_1���������
�2�
b__inference_score_convolutional_neural_network_1_layer_call_and_return_conditional_losses_51522202�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"
input_1���������
�2�
G__inference_score_convolutional_neural_network_1_layer_call_fn_51522224�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"
input_1���������
�2�
F__inference_conv2d_1_layer_call_and_return_conditional_losses_51522304�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_conv2d_1_layer_call_fn_51522313�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_flatten_1_layer_call_and_return_conditional_losses_51522319�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_flatten_1_layer_call_fn_51522324�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_3_layer_call_and_return_conditional_losses_51522335�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_3_layer_call_fn_51522344�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_4_layer_call_and_return_conditional_losses_51522355�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_4_layer_call_fn_51522364�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_5_layer_call_and_return_conditional_losses_51522374�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_5_layer_call_fn_51522383�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_51522293input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
#__inference__wrapped_model_51522119u!"4�1
*�'
%�"
input_1���������
� "3�0
.
output_1"�
output_1����������
F__inference_conv2d_1_layer_call_and_return_conditional_losses_51522304m7�4
-�*
(�%
inputs���������
� ".�+
$�!
0����������
� �
+__inference_conv2d_1_layer_call_fn_51522313`7�4
-�*
(�%
inputs���������
� "!������������
E__inference_dense_3_layer_call_and_return_conditional_losses_51522335]0�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� ~
*__inference_dense_3_layer_call_fn_51522344P0�-
&�#
!�
inputs����������
� "����������d�
E__inference_dense_4_layer_call_and_return_conditional_losses_51522355\/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� }
*__inference_dense_4_layer_call_fn_51522364O/�,
%�"
 �
inputs���������d
� "����������d�
E__inference_dense_5_layer_call_and_return_conditional_losses_51522374\!"/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� }
*__inference_dense_5_layer_call_fn_51522383O!"/�,
%�"
 �
inputs���������d
� "�����������
G__inference_flatten_1_layer_call_and_return_conditional_losses_51522319b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
,__inference_flatten_1_layer_call_fn_51522324U8�5
.�+
)�&
inputs����������
� "������������
b__inference_score_convolutional_neural_network_1_layer_call_and_return_conditional_losses_51522202g!"4�1
*�'
%�"
input_1���������
� "%�"
�
0���������
� �
G__inference_score_convolutional_neural_network_1_layer_call_fn_51522224Z!"4�1
*�'
%�"
input_1���������
� "�����������
&__inference_signature_wrapper_51522293�!"?�<
� 
5�2
0
input_1%�"
input_1���������"3�0
.
output_1"�
output_1���������