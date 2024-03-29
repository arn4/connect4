ܜ
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
0score_convolutional_neural_network/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*A
shared_name20score_convolutional_neural_network/conv2d/kernel
�
Dscore_convolutional_neural_network/conv2d/kernel/Read/ReadVariableOpReadVariableOp0score_convolutional_neural_network/conv2d/kernel*'
_output_shapes
:�*
dtype0
�
.score_convolutional_neural_network/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*?
shared_name0.score_convolutional_neural_network/conv2d/bias
�
Bscore_convolutional_neural_network/conv2d/bias/Read/ReadVariableOpReadVariableOp.score_convolutional_neural_network/conv2d/bias*
_output_shapes	
:�*
dtype0
�
/score_convolutional_neural_network/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*@
shared_name1/score_convolutional_neural_network/dense/kernel
�
Cscore_convolutional_neural_network/dense/kernel/Read/ReadVariableOpReadVariableOp/score_convolutional_neural_network/dense/kernel*
_output_shapes
:	�d*
dtype0
�
-score_convolutional_neural_network/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*>
shared_name/-score_convolutional_neural_network/dense/bias
�
Ascore_convolutional_neural_network/dense/bias/Read/ReadVariableOpReadVariableOp-score_convolutional_neural_network/dense/bias*
_output_shapes
:d*
dtype0
�
1score_convolutional_neural_network/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*B
shared_name31score_convolutional_neural_network/dense_1/kernel
�
Escore_convolutional_neural_network/dense_1/kernel/Read/ReadVariableOpReadVariableOp1score_convolutional_neural_network/dense_1/kernel*
_output_shapes

:dd*
dtype0
�
/score_convolutional_neural_network/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*@
shared_name1/score_convolutional_neural_network/dense_1/bias
�
Cscore_convolutional_neural_network/dense_1/bias/Read/ReadVariableOpReadVariableOp/score_convolutional_neural_network/dense_1/bias*
_output_shapes
:d*
dtype0
�
1score_convolutional_neural_network/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*B
shared_name31score_convolutional_neural_network/dense_2/kernel
�
Escore_convolutional_neural_network/dense_2/kernel/Read/ReadVariableOpReadVariableOp1score_convolutional_neural_network/dense_2/kernel*
_output_shapes

:d*
dtype0
�
/score_convolutional_neural_network/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/score_convolutional_neural_network/dense_2/bias
�
Cscore_convolutional_neural_network/dense_2/bias/Read/ReadVariableOpReadVariableOp/score_convolutional_neural_network/dense_2/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
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
regularization_losses
trainable_variables
	variables
		keras_api


signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
h

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
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
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
regularization_losses
trainable_variables
	variables
 
lj
VARIABLE_VALUE0score_convolutional_neural_network/conv2d/kernel&conv/kernel/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE.score_convolutional_neural_network/conv2d/bias$conv/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
regularization_losses
trainable_variables
	variables
 
 
 
�
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
regularization_losses
trainable_variables
	variables
nl
VARIABLE_VALUE/score_convolutional_neural_network/dense/kernel)hidden1/kernel/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-score_convolutional_neural_network/dense/bias'hidden1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
regularization_losses
trainable_variables
	variables
pn
VARIABLE_VALUE1score_convolutional_neural_network/dense_1/kernel)hidden2/kernel/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/score_convolutional_neural_network/dense_1/bias'hidden2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
regularization_losses
trainable_variables
	variables
qo
VARIABLE_VALUE1score_convolutional_neural_network/dense_2/kernel*outlayer/kernel/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE/score_convolutional_neural_network/dense_2/bias(outlayer/bias/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
#regularization_losses
$trainable_variables
%	variables
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10score_convolutional_neural_network/conv2d/kernel.score_convolutional_neural_network/conv2d/bias/score_convolutional_neural_network/dense/kernel-score_convolutional_neural_network/dense/bias1score_convolutional_neural_network/dense_1/kernel/score_convolutional_neural_network/dense_1/bias1score_convolutional_neural_network/dense_2/kernel/score_convolutional_neural_network/dense_2/bias*
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
GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_3452244
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameDscore_convolutional_neural_network/conv2d/kernel/Read/ReadVariableOpBscore_convolutional_neural_network/conv2d/bias/Read/ReadVariableOpCscore_convolutional_neural_network/dense/kernel/Read/ReadVariableOpAscore_convolutional_neural_network/dense/bias/Read/ReadVariableOpEscore_convolutional_neural_network/dense_1/kernel/Read/ReadVariableOpCscore_convolutional_neural_network/dense_1/bias/Read/ReadVariableOpEscore_convolutional_neural_network/dense_2/kernel/Read/ReadVariableOpCscore_convolutional_neural_network/dense_2/bias/Read/ReadVariableOpConst*
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
GPU 2J 8� *)
f$R"
 __inference__traced_save_3452381
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename0score_convolutional_neural_network/conv2d/kernel.score_convolutional_neural_network/conv2d/bias/score_convolutional_neural_network/dense/kernel-score_convolutional_neural_network/dense/bias1score_convolutional_neural_network/dense_1/kernel/score_convolutional_neural_network/dense_1/bias1score_convolutional_neural_network/dense_2/kernel/score_convolutional_neural_network/dense_2/bias*
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
GPU 2J 8� *,
f'R%
#__inference__traced_restore_3452415��
�
�
'__inference_dense_layer_call_fn_3452295

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
GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_34521132
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
�
%__inference_signature_wrapper_3452244
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
GPU 2J 8� *+
f&R$
"__inference__wrapped_model_34520702
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
E
)__inference_flatten_layer_call_fn_3452275

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
GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_34521002
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
�

�
B__inference_dense_layer_call_and_return_conditional_losses_3452286

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
�K
�

"__inference__wrapped_model_3452070
input_1c
Hscore_convolutional_neural_network_conv2d_conv2d_readvariableop_resource:�X
Iscore_convolutional_neural_network_conv2d_biasadd_readvariableop_resource:	�Z
Gscore_convolutional_neural_network_dense_matmul_readvariableop_resource:	�dV
Hscore_convolutional_neural_network_dense_biasadd_readvariableop_resource:d[
Iscore_convolutional_neural_network_dense_1_matmul_readvariableop_resource:ddX
Jscore_convolutional_neural_network_dense_1_biasadd_readvariableop_resource:d[
Iscore_convolutional_neural_network_dense_2_matmul_readvariableop_resource:dX
Jscore_convolutional_neural_network_dense_2_biasadd_readvariableop_resource:
identity��@score_convolutional_neural_network/conv2d/BiasAdd/ReadVariableOp�?score_convolutional_neural_network/conv2d/Conv2D/ReadVariableOp�?score_convolutional_neural_network/dense/BiasAdd/ReadVariableOp�>score_convolutional_neural_network/dense/MatMul/ReadVariableOp�Ascore_convolutional_neural_network/dense_1/BiasAdd/ReadVariableOp�@score_convolutional_neural_network/dense_1/MatMul/ReadVariableOp�Ascore_convolutional_neural_network/dense_2/BiasAdd/ReadVariableOp�@score_convolutional_neural_network/dense_2/MatMul/ReadVariableOp�
1score_convolutional_neural_network/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :23
1score_convolutional_neural_network/ExpandDims/dim�
-score_convolutional_neural_network/ExpandDims
ExpandDimsinput_1:score_convolutional_neural_network/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������2/
-score_convolutional_neural_network/ExpandDims�
'score_convolutional_neural_network/CastCast6score_convolutional_neural_network/ExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:���������2)
'score_convolutional_neural_network/Cast�
?score_convolutional_neural_network/conv2d/Conv2D/ReadVariableOpReadVariableOpHscore_convolutional_neural_network_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype02A
?score_convolutional_neural_network/conv2d/Conv2D/ReadVariableOp�
0score_convolutional_neural_network/conv2d/Conv2DConv2D+score_convolutional_neural_network/Cast:y:0Gscore_convolutional_neural_network/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
22
0score_convolutional_neural_network/conv2d/Conv2D�
@score_convolutional_neural_network/conv2d/BiasAdd/ReadVariableOpReadVariableOpIscore_convolutional_neural_network_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02B
@score_convolutional_neural_network/conv2d/BiasAdd/ReadVariableOp�
1score_convolutional_neural_network/conv2d/BiasAddBiasAdd9score_convolutional_neural_network/conv2d/Conv2D:output:0Hscore_convolutional_neural_network/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������23
1score_convolutional_neural_network/conv2d/BiasAdd�
.score_convolutional_neural_network/conv2d/ReluRelu:score_convolutional_neural_network/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:����������20
.score_convolutional_neural_network/conv2d/Relu�
0score_convolutional_neural_network/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����  22
0score_convolutional_neural_network/flatten/Const�
2score_convolutional_neural_network/flatten/ReshapeReshape<score_convolutional_neural_network/conv2d/Relu:activations:09score_convolutional_neural_network/flatten/Const:output:0*
T0*(
_output_shapes
:����������24
2score_convolutional_neural_network/flatten/Reshape�
>score_convolutional_neural_network/dense/MatMul/ReadVariableOpReadVariableOpGscore_convolutional_neural_network_dense_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02@
>score_convolutional_neural_network/dense/MatMul/ReadVariableOp�
/score_convolutional_neural_network/dense/MatMulMatMul;score_convolutional_neural_network/flatten/Reshape:output:0Fscore_convolutional_neural_network/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d21
/score_convolutional_neural_network/dense/MatMul�
?score_convolutional_neural_network/dense/BiasAdd/ReadVariableOpReadVariableOpHscore_convolutional_neural_network_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02A
?score_convolutional_neural_network/dense/BiasAdd/ReadVariableOp�
0score_convolutional_neural_network/dense/BiasAddBiasAdd9score_convolutional_neural_network/dense/MatMul:product:0Gscore_convolutional_neural_network/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d22
0score_convolutional_neural_network/dense/BiasAdd�
-score_convolutional_neural_network/dense/ReluRelu9score_convolutional_neural_network/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2/
-score_convolutional_neural_network/dense/Relu�
@score_convolutional_neural_network/dense_1/MatMul/ReadVariableOpReadVariableOpIscore_convolutional_neural_network_dense_1_matmul_readvariableop_resource*
_output_shapes

:dd*
dtype02B
@score_convolutional_neural_network/dense_1/MatMul/ReadVariableOp�
1score_convolutional_neural_network/dense_1/MatMulMatMul;score_convolutional_neural_network/dense/Relu:activations:0Hscore_convolutional_neural_network/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d23
1score_convolutional_neural_network/dense_1/MatMul�
Ascore_convolutional_neural_network/dense_1/BiasAdd/ReadVariableOpReadVariableOpJscore_convolutional_neural_network_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02C
Ascore_convolutional_neural_network/dense_1/BiasAdd/ReadVariableOp�
2score_convolutional_neural_network/dense_1/BiasAddBiasAdd;score_convolutional_neural_network/dense_1/MatMul:product:0Iscore_convolutional_neural_network/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d24
2score_convolutional_neural_network/dense_1/BiasAdd�
/score_convolutional_neural_network/dense_1/ReluRelu;score_convolutional_neural_network/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������d21
/score_convolutional_neural_network/dense_1/Relu�
@score_convolutional_neural_network/dense_2/MatMul/ReadVariableOpReadVariableOpIscore_convolutional_neural_network_dense_2_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02B
@score_convolutional_neural_network/dense_2/MatMul/ReadVariableOp�
1score_convolutional_neural_network/dense_2/MatMulMatMul=score_convolutional_neural_network/dense_1/Relu:activations:0Hscore_convolutional_neural_network/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������23
1score_convolutional_neural_network/dense_2/MatMul�
Ascore_convolutional_neural_network/dense_2/BiasAdd/ReadVariableOpReadVariableOpJscore_convolutional_neural_network_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02C
Ascore_convolutional_neural_network/dense_2/BiasAdd/ReadVariableOp�
2score_convolutional_neural_network/dense_2/BiasAddBiasAdd;score_convolutional_neural_network/dense_2/MatMul:product:0Iscore_convolutional_neural_network/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������24
2score_convolutional_neural_network/dense_2/BiasAdd�
IdentityIdentity;score_convolutional_neural_network/dense_2/BiasAdd:output:0A^score_convolutional_neural_network/conv2d/BiasAdd/ReadVariableOp@^score_convolutional_neural_network/conv2d/Conv2D/ReadVariableOp@^score_convolutional_neural_network/dense/BiasAdd/ReadVariableOp?^score_convolutional_neural_network/dense/MatMul/ReadVariableOpB^score_convolutional_neural_network/dense_1/BiasAdd/ReadVariableOpA^score_convolutional_neural_network/dense_1/MatMul/ReadVariableOpB^score_convolutional_neural_network/dense_2/BiasAdd/ReadVariableOpA^score_convolutional_neural_network/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2�
@score_convolutional_neural_network/conv2d/BiasAdd/ReadVariableOp@score_convolutional_neural_network/conv2d/BiasAdd/ReadVariableOp2�
?score_convolutional_neural_network/conv2d/Conv2D/ReadVariableOp?score_convolutional_neural_network/conv2d/Conv2D/ReadVariableOp2�
?score_convolutional_neural_network/dense/BiasAdd/ReadVariableOp?score_convolutional_neural_network/dense/BiasAdd/ReadVariableOp2�
>score_convolutional_neural_network/dense/MatMul/ReadVariableOp>score_convolutional_neural_network/dense/MatMul/ReadVariableOp2�
Ascore_convolutional_neural_network/dense_1/BiasAdd/ReadVariableOpAscore_convolutional_neural_network/dense_1/BiasAdd/ReadVariableOp2�
@score_convolutional_neural_network/dense_1/MatMul/ReadVariableOp@score_convolutional_neural_network/dense_1/MatMul/ReadVariableOp2�
Ascore_convolutional_neural_network/dense_2/BiasAdd/ReadVariableOpAscore_convolutional_neural_network/dense_2/BiasAdd/ReadVariableOp2�
@score_convolutional_neural_network/dense_2/MatMul/ReadVariableOp@score_convolutional_neural_network/dense_2/MatMul/ReadVariableOp:T P
+
_output_shapes
:���������
!
_user_specified_name	input_1
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_3452100

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
�
�
)__inference_dense_2_layer_call_fn_3452334

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
GPU 2J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_34521462
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
�

�
D__inference_dense_1_layer_call_and_return_conditional_losses_3452130

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
�	
�
D__inference_dense_2_layer_call_and_return_conditional_losses_3452146

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

�
D__inference_dense_1_layer_call_and_return_conditional_losses_3452306

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
�
�
C__inference_conv2d_layer_call_and_return_conditional_losses_3452255

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
�
�
)__inference_dense_1_layer_call_fn_3452315

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
GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_34521302
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
�
�
___inference_score_convolutional_neural_network_layer_call_and_return_conditional_losses_3452153
input_1)
conv2d_3452089:�
conv2d_3452091:	� 
dense_3452114:	�d
dense_3452116:d!
dense_1_3452131:dd
dense_1_3452133:d!
dense_2_3452147:d
dense_2_3452149:
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
conv2d/StatefulPartitionedCallStatefulPartitionedCallCast:y:0conv2d_3452089conv2d_3452091*
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
GPU 2J 8� *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_34520882 
conv2d/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_34521002
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_3452114dense_3452116*
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
GPU 2J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_34521132
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_3452131dense_1_3452133*
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
GPU 2J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_34521302!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_3452147dense_2_3452149*
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
GPU 2J 8� *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_34521462!
dense_2/StatefulPartitionedCall�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

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
�	
�
D__inference_dense_2_layer_call_and_return_conditional_losses_3452325

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
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_3452270

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
D__inference_score_convolutional_neural_network_layer_call_fn_3452175
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
GPU 2J 8� *h
fcRa
___inference_score_convolutional_neural_network_layer_call_and_return_conditional_losses_34521532
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
�"
�
 __inference__traced_save_3452381
file_prefixO
Ksavev2_score_convolutional_neural_network_conv2d_kernel_read_readvariableopM
Isavev2_score_convolutional_neural_network_conv2d_bias_read_readvariableopN
Jsavev2_score_convolutional_neural_network_dense_kernel_read_readvariableopL
Hsavev2_score_convolutional_neural_network_dense_bias_read_readvariableopP
Lsavev2_score_convolutional_neural_network_dense_1_kernel_read_readvariableopN
Jsavev2_score_convolutional_neural_network_dense_1_bias_read_readvariableopP
Lsavev2_score_convolutional_neural_network_dense_2_kernel_read_readvariableopN
Jsavev2_score_convolutional_neural_network_dense_2_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Ksavev2_score_convolutional_neural_network_conv2d_kernel_read_readvariableopIsavev2_score_convolutional_neural_network_conv2d_bias_read_readvariableopJsavev2_score_convolutional_neural_network_dense_kernel_read_readvariableopHsavev2_score_convolutional_neural_network_dense_bias_read_readvariableopLsavev2_score_convolutional_neural_network_dense_1_kernel_read_readvariableopJsavev2_score_convolutional_neural_network_dense_1_bias_read_readvariableopLsavev2_score_convolutional_neural_network_dense_2_kernel_read_readvariableopJsavev2_score_convolutional_neural_network_dense_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
B__inference_dense_layer_call_and_return_conditional_losses_3452113

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
�
�
C__inference_conv2d_layer_call_and_return_conditional_losses_3452088

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
�)
�
#__inference__traced_restore_3452415
file_prefix\
Aassignvariableop_score_convolutional_neural_network_conv2d_kernel:�P
Aassignvariableop_1_score_convolutional_neural_network_conv2d_bias:	�U
Bassignvariableop_2_score_convolutional_neural_network_dense_kernel:	�dN
@assignvariableop_3_score_convolutional_neural_network_dense_bias:dV
Dassignvariableop_4_score_convolutional_neural_network_dense_1_kernel:ddP
Bassignvariableop_5_score_convolutional_neural_network_dense_1_bias:dV
Dassignvariableop_6_score_convolutional_neural_network_dense_2_kernel:dP
Bassignvariableop_7_score_convolutional_neural_network_dense_2_bias:

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
AssignVariableOpAssignVariableOpAassignvariableop_score_convolutional_neural_network_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpAassignvariableop_1_score_convolutional_neural_network_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpBassignvariableop_2_score_convolutional_neural_network_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp@assignvariableop_3_score_convolutional_neural_network_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpDassignvariableop_4_score_convolutional_neural_network_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpBassignvariableop_5_score_convolutional_neural_network_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpDassignvariableop_6_score_convolutional_neural_network_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpBassignvariableop_7_score_convolutional_neural_network_dense_2_biasIdentity_7:output:0"/device:CPU:0*
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
�
�
(__inference_conv2d_layer_call_fn_3452264

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
GPU 2J 8� *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_34520882
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
regularization_losses
trainable_variables
	variables
		keras_api


signatures
*E&call_and_return_all_conditional_losses
F_default_save_signature
G__call__"�
_tf_keras_model�{"name": "score_convolutional_neural_network", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "ScoreConvolutionalNeuralNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [204, 6, 7]}, "int8", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "ScoreConvolutionalNeuralNetwork"}}
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*H&call_and_return_all_conditional_losses
I__call__"�

_tf_keras_layer�
{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 7, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 7, 1]}, "dtype": "float32", "filters": 150, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [204, 6, 7, 1]}}
�
regularization_losses
trainable_variables
	variables
	keras_api
*J&call_and_return_all_conditional_losses
K__call__"�
_tf_keras_layer�{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 5}}
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*L&call_and_return_all_conditional_losses
M__call__"�
_tf_keras_layer�{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1800}}, "shared_object_id": 9}, "build_input_shape": {"class_name": "TensorShape", "items": [204, 1800]}}
�

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
*N&call_and_return_all_conditional_losses
O__call__"�
_tf_keras_layer�{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 13}, "build_input_shape": {"class_name": "TensorShape", "items": [204, 100]}}
�

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
*P&call_and_return_all_conditional_losses
Q__call__"�
_tf_keras_layer�{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [204, 100]}}
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
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
regularization_losses
trainable_variables
	variables
G__call__
F_default_save_signature
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
,
Rserving_default"
signature_map
K:I�20score_convolutional_neural_network/conv2d/kernel
=:;�2.score_convolutional_neural_network/conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
regularization_losses
trainable_variables
	variables
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
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
regularization_losses
trainable_variables
	variables
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
B:@	�d2/score_convolutional_neural_network/dense/kernel
;:9d2-score_convolutional_neural_network/dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
regularization_losses
trainable_variables
	variables
M__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
C:Add21score_convolutional_neural_network/dense_1/kernel
=:;d2/score_convolutional_neural_network/dense_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
regularization_losses
trainable_variables
	variables
O__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
C:Ad21score_convolutional_neural_network/dense_2/kernel
=:;2/score_convolutional_neural_network/dense_2/bias
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
#regularization_losses
$trainable_variables
%	variables
Q__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_dict_wrapper
�2�
___inference_score_convolutional_neural_network_layer_call_and_return_conditional_losses_3452153�
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
"__inference__wrapped_model_3452070�
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
D__inference_score_convolutional_neural_network_layer_call_fn_3452175�
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
C__inference_conv2d_layer_call_and_return_conditional_losses_3452255�
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
(__inference_conv2d_layer_call_fn_3452264�
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
D__inference_flatten_layer_call_and_return_conditional_losses_3452270�
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
)__inference_flatten_layer_call_fn_3452275�
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
B__inference_dense_layer_call_and_return_conditional_losses_3452286�
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
'__inference_dense_layer_call_fn_3452295�
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
D__inference_dense_1_layer_call_and_return_conditional_losses_3452306�
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
)__inference_dense_1_layer_call_fn_3452315�
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
D__inference_dense_2_layer_call_and_return_conditional_losses_3452325�
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
)__inference_dense_2_layer_call_fn_3452334�
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
%__inference_signature_wrapper_3452244input_1"�
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
"__inference__wrapped_model_3452070u!"4�1
*�'
%�"
input_1���������
� "3�0
.
output_1"�
output_1����������
C__inference_conv2d_layer_call_and_return_conditional_losses_3452255m7�4
-�*
(�%
inputs���������
� ".�+
$�!
0����������
� �
(__inference_conv2d_layer_call_fn_3452264`7�4
-�*
(�%
inputs���������
� "!������������
D__inference_dense_1_layer_call_and_return_conditional_losses_3452306\/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� |
)__inference_dense_1_layer_call_fn_3452315O/�,
%�"
 �
inputs���������d
� "����������d�
D__inference_dense_2_layer_call_and_return_conditional_losses_3452325\!"/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� |
)__inference_dense_2_layer_call_fn_3452334O!"/�,
%�"
 �
inputs���������d
� "�����������
B__inference_dense_layer_call_and_return_conditional_losses_3452286]0�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� {
'__inference_dense_layer_call_fn_3452295P0�-
&�#
!�
inputs����������
� "����������d�
D__inference_flatten_layer_call_and_return_conditional_losses_3452270b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
)__inference_flatten_layer_call_fn_3452275U8�5
.�+
)�&
inputs����������
� "������������
___inference_score_convolutional_neural_network_layer_call_and_return_conditional_losses_3452153g!"4�1
*�'
%�"
input_1���������
� "%�"
�
0���������
� �
D__inference_score_convolutional_neural_network_layer_call_fn_3452175Z!"4�1
*�'
%�"
input_1���������
� "�����������
%__inference_signature_wrapper_3452244�!"?�<
� 
5�2
0
input_1%�"
input_1���������"3�0
.
output_1"�
output_1���������