��
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
 �"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718��
�
*convolutional_neural_network/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*;
shared_name,*convolutional_neural_network/conv2d/kernel
�
>convolutional_neural_network/conv2d/kernel/Read/ReadVariableOpReadVariableOp*convolutional_neural_network/conv2d/kernel*&
_output_shapes
:d*
dtype0
�
(convolutional_neural_network/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*9
shared_name*(convolutional_neural_network/conv2d/bias
�
<convolutional_neural_network/conv2d/bias/Read/ReadVariableOpReadVariableOp(convolutional_neural_network/conv2d/bias*
_output_shapes
:d*
dtype0
�
)convolutional_neural_network/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�	2*:
shared_name+)convolutional_neural_network/dense/kernel
�
=convolutional_neural_network/dense/kernel/Read/ReadVariableOpReadVariableOp)convolutional_neural_network/dense/kernel*
_output_shapes
:	�	2*
dtype0
�
'convolutional_neural_network/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*8
shared_name)'convolutional_neural_network/dense/bias
�
;convolutional_neural_network/dense/bias/Read/ReadVariableOpReadVariableOp'convolutional_neural_network/dense/bias*
_output_shapes
:2*
dtype0
�
+convolutional_neural_network/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*<
shared_name-+convolutional_neural_network/dense_1/kernel
�
?convolutional_neural_network/dense_1/kernel/Read/ReadVariableOpReadVariableOp+convolutional_neural_network/dense_1/kernel*
_output_shapes

:22*
dtype0
�
)convolutional_neural_network/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*:
shared_name+)convolutional_neural_network/dense_1/bias
�
=convolutional_neural_network/dense_1/bias/Read/ReadVariableOpReadVariableOp)convolutional_neural_network/dense_1/bias*
_output_shapes
:2*
dtype0
�
+convolutional_neural_network/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*<
shared_name-+convolutional_neural_network/dense_2/kernel
�
?convolutional_neural_network/dense_2/kernel/Read/ReadVariableOpReadVariableOp+convolutional_neural_network/dense_2/kernel*
_output_shapes

:2*
dtype0
�
)convolutional_neural_network/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)convolutional_neural_network/dense_2/bias
�
=convolutional_neural_network/dense_2/bias/Read/ReadVariableOpReadVariableOp)convolutional_neural_network/dense_2/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
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
'layer_regularization_losses
(layer_metrics
trainable_variables
)metrics
*non_trainable_variables

+layers
 
fd
VARIABLE_VALUE*convolutional_neural_network/conv2d/kernel&conv/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE(convolutional_neural_network/conv2d/bias$conv/bias/.ATTRIBUTES/VARIABLE_VALUE
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
,layer_regularization_losses
-layer_metrics
trainable_variables
.metrics
/non_trainable_variables

0layers
 
 
 
�
	variables
regularization_losses
1layer_regularization_losses
2layer_metrics
trainable_variables
3metrics
4non_trainable_variables

5layers
hf
VARIABLE_VALUE)convolutional_neural_network/dense/kernel)hidden1/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'convolutional_neural_network/dense/bias'hidden1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
6layer_regularization_losses
7layer_metrics
trainable_variables
8metrics
9non_trainable_variables

:layers
jh
VARIABLE_VALUE+convolutional_neural_network/dense_1/kernel)hidden2/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)convolutional_neural_network/dense_1/bias'hidden2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
;layer_regularization_losses
<layer_metrics
trainable_variables
=metrics
>non_trainable_variables

?layers
ki
VARIABLE_VALUE+convolutional_neural_network/dense_2/kernel*outlayer/kernel/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE)convolutional_neural_network/dense_2/bias(outlayer/bias/.ATTRIBUTES/VARIABLE_VALUE
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
@layer_regularization_losses
Alayer_metrics
%trainable_variables
Bmetrics
Cnon_trainable_variables

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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1*convolutional_neural_network/conv2d/kernel(convolutional_neural_network/conv2d/bias)convolutional_neural_network/dense/kernel'convolutional_neural_network/dense/bias+convolutional_neural_network/dense_1/kernel)convolutional_neural_network/dense_1/bias+convolutional_neural_network/dense_2/kernel)convolutional_neural_network/dense_2/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_59248
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename>convolutional_neural_network/conv2d/kernel/Read/ReadVariableOp<convolutional_neural_network/conv2d/bias/Read/ReadVariableOp=convolutional_neural_network/dense/kernel/Read/ReadVariableOp;convolutional_neural_network/dense/bias/Read/ReadVariableOp?convolutional_neural_network/dense_1/kernel/Read/ReadVariableOp=convolutional_neural_network/dense_1/bias/Read/ReadVariableOp?convolutional_neural_network/dense_2/kernel/Read/ReadVariableOp=convolutional_neural_network/dense_2/bias/Read/ReadVariableOpConst*
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
GPU 2J 8� *'
f"R 
__inference__traced_save_59385
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename*convolutional_neural_network/conv2d/kernel(convolutional_neural_network/conv2d/bias)convolutional_neural_network/dense/kernel'convolutional_neural_network/dense/bias+convolutional_neural_network/dense_1/kernel)convolutional_neural_network/dense_1/bias+convolutional_neural_network/dense_2/kernel)convolutional_neural_network/dense_2/bias*
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
GPU 2J 8� **
f%R#
!__inference__traced_restore_59419��
�
�
W__inference_convolutional_neural_network_layer_call_and_return_conditional_losses_59157
input_1&
conv2d_59093:d
conv2d_59095:d
dense_59118:	�	2
dense_59120:2
dense_1_59135:22
dense_1_59137:2
dense_2_59151:2
dense_2_59153:
identity��conv2d/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCallb
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
conv2d/StatefulPartitionedCallStatefulPartitionedCallCast:y:0conv2d_59093conv2d_59095*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_590922 
conv2d/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_591042
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_59118dense_59120*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_591172
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_59135dense_1_59137*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_591342!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_59151dense_2_59153*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_591502!
dense_2/StatefulPartitionedCall�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
'__inference_dense_2_layer_call_fn_59328

inputs
unknown:2
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_591502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�)
�
!__inference__traced_restore_59419
file_prefixU
;assignvariableop_convolutional_neural_network_conv2d_kernel:dI
;assignvariableop_1_convolutional_neural_network_conv2d_bias:dO
<assignvariableop_2_convolutional_neural_network_dense_kernel:	�	2H
:assignvariableop_3_convolutional_neural_network_dense_bias:2P
>assignvariableop_4_convolutional_neural_network_dense_1_kernel:22J
<assignvariableop_5_convolutional_neural_network_dense_1_bias:2P
>assignvariableop_6_convolutional_neural_network_dense_2_kernel:2J
<assignvariableop_7_convolutional_neural_network_dense_2_bias:

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
AssignVariableOpAssignVariableOp;assignvariableop_convolutional_neural_network_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp;assignvariableop_1_convolutional_neural_network_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp<assignvariableop_2_convolutional_neural_network_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp:assignvariableop_3_convolutional_neural_network_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp>assignvariableop_4_convolutional_neural_network_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp<assignvariableop_5_convolutional_neural_network_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp>assignvariableop_6_convolutional_neural_network_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp<assignvariableop_7_convolutional_neural_network_dense_2_biasIdentity_7:output:0"/device:CPU:0*
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
B__inference_dense_1_layer_call_and_return_conditional_losses_59319

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������22
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
C
'__inference_flatten_layer_call_fn_59273

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
:����������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_591042
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_59279

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�

�
B__inference_dense_1_layer_call_and_return_conditional_losses_59134

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������22
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�E
�	
 __inference__wrapped_model_59074
input_1\
Bconvolutional_neural_network_conv2d_conv2d_readvariableop_resource:dQ
Cconvolutional_neural_network_conv2d_biasadd_readvariableop_resource:dT
Aconvolutional_neural_network_dense_matmul_readvariableop_resource:	�	2P
Bconvolutional_neural_network_dense_biasadd_readvariableop_resource:2U
Cconvolutional_neural_network_dense_1_matmul_readvariableop_resource:22R
Dconvolutional_neural_network_dense_1_biasadd_readvariableop_resource:2U
Cconvolutional_neural_network_dense_2_matmul_readvariableop_resource:2R
Dconvolutional_neural_network_dense_2_biasadd_readvariableop_resource:
identity��:convolutional_neural_network/conv2d/BiasAdd/ReadVariableOp�9convolutional_neural_network/conv2d/Conv2D/ReadVariableOp�9convolutional_neural_network/dense/BiasAdd/ReadVariableOp�8convolutional_neural_network/dense/MatMul/ReadVariableOp�;convolutional_neural_network/dense_1/BiasAdd/ReadVariableOp�:convolutional_neural_network/dense_1/MatMul/ReadVariableOp�;convolutional_neural_network/dense_2/BiasAdd/ReadVariableOp�:convolutional_neural_network/dense_2/MatMul/ReadVariableOp�
+convolutional_neural_network/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+convolutional_neural_network/ExpandDims/dim�
'convolutional_neural_network/ExpandDims
ExpandDimsinput_14convolutional_neural_network/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2)
'convolutional_neural_network/ExpandDims�
!convolutional_neural_network/CastCast0convolutional_neural_network/ExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:���������2#
!convolutional_neural_network/Cast�
9convolutional_neural_network/conv2d/Conv2D/ReadVariableOpReadVariableOpBconvolutional_neural_network_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:d*
dtype02;
9convolutional_neural_network/conv2d/Conv2D/ReadVariableOp�
*convolutional_neural_network/conv2d/Conv2DConv2D%convolutional_neural_network/Cast:y:0Aconvolutional_neural_network/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������d*
paddingVALID*
strides
2,
*convolutional_neural_network/conv2d/Conv2D�
:convolutional_neural_network/conv2d/BiasAdd/ReadVariableOpReadVariableOpCconvolutional_neural_network_conv2d_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02<
:convolutional_neural_network/conv2d/BiasAdd/ReadVariableOp�
+convolutional_neural_network/conv2d/BiasAddBiasAdd3convolutional_neural_network/conv2d/Conv2D:output:0Bconvolutional_neural_network/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������d2-
+convolutional_neural_network/conv2d/BiasAdd�
(convolutional_neural_network/conv2d/ReluRelu4convolutional_neural_network/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������d2*
(convolutional_neural_network/conv2d/Relu�
*convolutional_neural_network/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2,
*convolutional_neural_network/flatten/Const�
,convolutional_neural_network/flatten/ReshapeReshape6convolutional_neural_network/conv2d/Relu:activations:03convolutional_neural_network/flatten/Const:output:0*
T0*(
_output_shapes
:����������	2.
,convolutional_neural_network/flatten/Reshape�
8convolutional_neural_network/dense/MatMul/ReadVariableOpReadVariableOpAconvolutional_neural_network_dense_matmul_readvariableop_resource*
_output_shapes
:	�	2*
dtype02:
8convolutional_neural_network/dense/MatMul/ReadVariableOp�
)convolutional_neural_network/dense/MatMulMatMul5convolutional_neural_network/flatten/Reshape:output:0@convolutional_neural_network/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22+
)convolutional_neural_network/dense/MatMul�
9convolutional_neural_network/dense/BiasAdd/ReadVariableOpReadVariableOpBconvolutional_neural_network_dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02;
9convolutional_neural_network/dense/BiasAdd/ReadVariableOp�
*convolutional_neural_network/dense/BiasAddBiasAdd3convolutional_neural_network/dense/MatMul:product:0Aconvolutional_neural_network/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22,
*convolutional_neural_network/dense/BiasAdd�
'convolutional_neural_network/dense/ReluRelu3convolutional_neural_network/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������22)
'convolutional_neural_network/dense/Relu�
:convolutional_neural_network/dense_1/MatMul/ReadVariableOpReadVariableOpCconvolutional_neural_network_dense_1_matmul_readvariableop_resource*
_output_shapes

:22*
dtype02<
:convolutional_neural_network/dense_1/MatMul/ReadVariableOp�
+convolutional_neural_network/dense_1/MatMulMatMul5convolutional_neural_network/dense/Relu:activations:0Bconvolutional_neural_network/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22-
+convolutional_neural_network/dense_1/MatMul�
;convolutional_neural_network/dense_1/BiasAdd/ReadVariableOpReadVariableOpDconvolutional_neural_network_dense_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02=
;convolutional_neural_network/dense_1/BiasAdd/ReadVariableOp�
,convolutional_neural_network/dense_1/BiasAddBiasAdd5convolutional_neural_network/dense_1/MatMul:product:0Cconvolutional_neural_network/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22.
,convolutional_neural_network/dense_1/BiasAdd�
)convolutional_neural_network/dense_1/ReluRelu5convolutional_neural_network/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������22+
)convolutional_neural_network/dense_1/Relu�
:convolutional_neural_network/dense_2/MatMul/ReadVariableOpReadVariableOpCconvolutional_neural_network_dense_2_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02<
:convolutional_neural_network/dense_2/MatMul/ReadVariableOp�
+convolutional_neural_network/dense_2/MatMulMatMul7convolutional_neural_network/dense_1/Relu:activations:0Bconvolutional_neural_network/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2-
+convolutional_neural_network/dense_2/MatMul�
;convolutional_neural_network/dense_2/BiasAdd/ReadVariableOpReadVariableOpDconvolutional_neural_network_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;convolutional_neural_network/dense_2/BiasAdd/ReadVariableOp�
,convolutional_neural_network/dense_2/BiasAddBiasAdd5convolutional_neural_network/dense_2/MatMul:product:0Cconvolutional_neural_network/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2.
,convolutional_neural_network/dense_2/BiasAdd�
IdentityIdentity5convolutional_neural_network/dense_2/BiasAdd:output:0;^convolutional_neural_network/conv2d/BiasAdd/ReadVariableOp:^convolutional_neural_network/conv2d/Conv2D/ReadVariableOp:^convolutional_neural_network/dense/BiasAdd/ReadVariableOp9^convolutional_neural_network/dense/MatMul/ReadVariableOp<^convolutional_neural_network/dense_1/BiasAdd/ReadVariableOp;^convolutional_neural_network/dense_1/MatMul/ReadVariableOp<^convolutional_neural_network/dense_2/BiasAdd/ReadVariableOp;^convolutional_neural_network/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2x
:convolutional_neural_network/conv2d/BiasAdd/ReadVariableOp:convolutional_neural_network/conv2d/BiasAdd/ReadVariableOp2v
9convolutional_neural_network/conv2d/Conv2D/ReadVariableOp9convolutional_neural_network/conv2d/Conv2D/ReadVariableOp2v
9convolutional_neural_network/dense/BiasAdd/ReadVariableOp9convolutional_neural_network/dense/BiasAdd/ReadVariableOp2t
8convolutional_neural_network/dense/MatMul/ReadVariableOp8convolutional_neural_network/dense/MatMul/ReadVariableOp2z
;convolutional_neural_network/dense_1/BiasAdd/ReadVariableOp;convolutional_neural_network/dense_1/BiasAdd/ReadVariableOp2x
:convolutional_neural_network/dense_1/MatMul/ReadVariableOp:convolutional_neural_network/dense_1/MatMul/ReadVariableOp2z
;convolutional_neural_network/dense_2/BiasAdd/ReadVariableOp;convolutional_neural_network/dense_2/BiasAdd/ReadVariableOp2x
:convolutional_neural_network/dense_2/MatMul/ReadVariableOp:convolutional_neural_network/dense_2/MatMul/ReadVariableOp:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�	
�
#__inference_signature_wrapper_59248
input_1!
unknown:d
	unknown_0:d
	unknown_1:	�	2
	unknown_2:2
	unknown_3:22
	unknown_4:2
	unknown_5:2
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_590742
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

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
�
�
&__inference_conv2d_layer_call_fn_59257

inputs!
unknown:d
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_590922
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������d2

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
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_59104

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d:W S
/
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
A__inference_conv2d_layer_call_and_return_conditional_losses_59268

inputs8
conv2d_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:d*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������d*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������d2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������d2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������d2

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
�
�
%__inference_dense_layer_call_fn_59288

inputs
unknown:	�	2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_591172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�	
�
B__inference_dense_2_layer_call_and_return_conditional_losses_59338

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�	
�
<__inference_convolutional_neural_network_layer_call_fn_59179
input_1!
unknown:d
	unknown_0:d
	unknown_1:	�	2
	unknown_2:2
	unknown_3:22
	unknown_4:2
	unknown_5:2
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *`
f[RY
W__inference_convolutional_neural_network_layer_call_and_return_conditional_losses_591572
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

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
�

�
@__inference_dense_layer_call_and_return_conditional_losses_59117

inputs1
matmul_readvariableop_resource:	�	2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�	2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������22
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�!
�
__inference__traced_save_59385
file_prefixI
Esavev2_convolutional_neural_network_conv2d_kernel_read_readvariableopG
Csavev2_convolutional_neural_network_conv2d_bias_read_readvariableopH
Dsavev2_convolutional_neural_network_dense_kernel_read_readvariableopF
Bsavev2_convolutional_neural_network_dense_bias_read_readvariableopJ
Fsavev2_convolutional_neural_network_dense_1_kernel_read_readvariableopH
Dsavev2_convolutional_neural_network_dense_1_bias_read_readvariableopJ
Fsavev2_convolutional_neural_network_dense_2_kernel_read_readvariableopH
Dsavev2_convolutional_neural_network_dense_2_bias_read_readvariableop
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
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Esavev2_convolutional_neural_network_conv2d_kernel_read_readvariableopCsavev2_convolutional_neural_network_conv2d_bias_read_readvariableopDsavev2_convolutional_neural_network_dense_kernel_read_readvariableopBsavev2_convolutional_neural_network_dense_bias_read_readvariableopFsavev2_convolutional_neural_network_dense_1_kernel_read_readvariableopDsavev2_convolutional_neural_network_dense_1_bias_read_readvariableopFsavev2_convolutional_neural_network_dense_2_kernel_read_readvariableopDsavev2_convolutional_neural_network_dense_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*`
_input_shapesO
M: :d:d:	�	2:2:22:2:2:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:d: 

_output_shapes
:d:%!

_output_shapes
:	�	2: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::	

_output_shapes
: 
�
�
'__inference_dense_1_layer_call_fn_59308

inputs
unknown:22
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_591342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
A__inference_conv2d_layer_call_and_return_conditional_losses_59092

inputs8
conv2d_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:d*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������d*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������d2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������d2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������d2

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
@__inference_dense_layer_call_and_return_conditional_losses_59299

inputs1
matmul_readvariableop_resource:	�	2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�	2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������22
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�	
�
B__inference_dense_2_layer_call_and_return_conditional_losses_59150

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
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
StatefulPartitionedCall:0���������tensorflow/serving/predict:�|
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
E__call__
*F&call_and_return_all_conditional_losses
G_default_save_signature"�
_tf_keras_model�{"name": "convolutional_neural_network", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "ConvolutionalNeuralNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1000, 6, 7]}, "int8", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "ConvolutionalNeuralNetwork"}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
H__call__
*I&call_and_return_all_conditional_losses"�

_tf_keras_layer�
{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 7, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 7, 1]}, "dtype": "float32", "filters": 100, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1000, 6, 7, 1]}}
�
	variables
regularization_losses
trainable_variables
	keras_api
J__call__
*K&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 5}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
L__call__
*M&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1200}}, "shared_object_id": 9}, "build_input_shape": {"class_name": "TensorShape", "items": [1000, 1200]}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
N__call__
*O&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 13}, "build_input_shape": {"class_name": "TensorShape", "items": [1000, 50]}}
�

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 7, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [1000, 50]}}
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
'layer_regularization_losses
(layer_metrics
trainable_variables
)metrics
*non_trainable_variables

+layers
E__call__
G_default_save_signature
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
,
Rserving_default"
signature_map
D:Bd2*convolutional_neural_network/conv2d/kernel
6:4d2(convolutional_neural_network/conv2d/bias
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
,layer_regularization_losses
-layer_metrics
trainable_variables
.metrics
/non_trainable_variables

0layers
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
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
1layer_regularization_losses
2layer_metrics
trainable_variables
3metrics
4non_trainable_variables

5layers
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
<::	�	22)convolutional_neural_network/dense/kernel
5:322'convolutional_neural_network/dense/bias
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
6layer_regularization_losses
7layer_metrics
trainable_variables
8metrics
9non_trainable_variables

:layers
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
=:;222+convolutional_neural_network/dense_1/kernel
7:522)convolutional_neural_network/dense_1/bias
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
;layer_regularization_losses
<layer_metrics
trainable_variables
=metrics
>non_trainable_variables

?layers
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
=:;22+convolutional_neural_network/dense_2/kernel
7:52)convolutional_neural_network/dense_2/bias
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
@layer_regularization_losses
Alayer_metrics
%trainable_variables
Bmetrics
Cnon_trainable_variables

Dlayers
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
�2�
<__inference_convolutional_neural_network_layer_call_fn_59179�
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
W__inference_convolutional_neural_network_layer_call_and_return_conditional_losses_59157�
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
 __inference__wrapped_model_59074�
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
�2�
&__inference_conv2d_layer_call_fn_59257�
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
A__inference_conv2d_layer_call_and_return_conditional_losses_59268�
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
'__inference_flatten_layer_call_fn_59273�
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
B__inference_flatten_layer_call_and_return_conditional_losses_59279�
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
%__inference_dense_layer_call_fn_59288�
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
@__inference_dense_layer_call_and_return_conditional_losses_59299�
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
'__inference_dense_1_layer_call_fn_59308�
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
B__inference_dense_1_layer_call_and_return_conditional_losses_59319�
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
'__inference_dense_2_layer_call_fn_59328�
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
B__inference_dense_2_layer_call_and_return_conditional_losses_59338�
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
#__inference_signature_wrapper_59248input_1"�
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
 __inference__wrapped_model_59074u!"4�1
*�'
%�"
input_1���������
� "3�0
.
output_1"�
output_1����������
A__inference_conv2d_layer_call_and_return_conditional_losses_59268l7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������d
� �
&__inference_conv2d_layer_call_fn_59257_7�4
-�*
(�%
inputs���������
� " ����������d�
W__inference_convolutional_neural_network_layer_call_and_return_conditional_losses_59157g!"4�1
*�'
%�"
input_1���������
� "%�"
�
0���������
� �
<__inference_convolutional_neural_network_layer_call_fn_59179Z!"4�1
*�'
%�"
input_1���������
� "�����������
B__inference_dense_1_layer_call_and_return_conditional_losses_59319\/�,
%�"
 �
inputs���������2
� "%�"
�
0���������2
� z
'__inference_dense_1_layer_call_fn_59308O/�,
%�"
 �
inputs���������2
� "����������2�
B__inference_dense_2_layer_call_and_return_conditional_losses_59338\!"/�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� z
'__inference_dense_2_layer_call_fn_59328O!"/�,
%�"
 �
inputs���������2
� "�����������
@__inference_dense_layer_call_and_return_conditional_losses_59299]0�-
&�#
!�
inputs����������	
� "%�"
�
0���������2
� y
%__inference_dense_layer_call_fn_59288P0�-
&�#
!�
inputs����������	
� "����������2�
B__inference_flatten_layer_call_and_return_conditional_losses_59279a7�4
-�*
(�%
inputs���������d
� "&�#
�
0����������	
� 
'__inference_flatten_layer_call_fn_59273T7�4
-�*
(�%
inputs���������d
� "�����������	�
#__inference_signature_wrapper_59248�!"?�<
� 
5�2
0
input_1%�"
input_1���������"3�0
.
output_1"�
output_1���������