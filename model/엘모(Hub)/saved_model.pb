ин
ф0ћ0
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeintѕ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
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
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
џ
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
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
i
	DecodeRaw	
bytes
output"out_type""
out_typetype:
2	
"
little_endianbool(
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

h
Equal
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(љ
)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
$

LogicalAnd
x

y

z
љ
!
LoopCond	
input


output

q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
ї
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
н
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
ї
Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	љ
2
NextIteration	
data"T
output"T"	
Ttype
k
NotEqual
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(љ
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
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	ѕ
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
Ц
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	ѕ
і
ReverseSequence

input"T
seq_lengths"Tlen
output"T"
seq_dimint"
	batch_dimint "	
Ttype"
Tlentype0	:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
╝
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Ш
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
c
StringSplit	
input
	delimiter
indices	

values	
shape	"

skip_emptybool(
:
Sub
x"T
y"T
z"T"
Ttype:
2	
ї
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:ѕ
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetypeѕ
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttypeѕ
я
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring ѕ
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttypeѕ
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ
9
VarIsInitializedOp
resource
is_initialized
ѕ*1.14.02unknown8јй
_
textPlaceholder*
shape:         *
dtype0*#
_output_shapes
:         
S
StringSplit/ConstConst*
dtype0*
_output_shapes
: *
value	B B 
}
StringSplit/StringSplitStringSplittextStringSplit/Const*<
_output_shapes*
(:         :         :
\
SparseToDense/default_valueConst*
valueB B *
dtype0*
_output_shapes
: 
Н
SparseToDenseSparseToDenseStringSplit/StringSplitStringSplit/StringSplit:2StringSplit/StringSplit:1SparseToDense/default_value*
Tindices0	*0
_output_shapes
:                  *
T0
K

NotEqual/yConst*
valueB B *
dtype0*
_output_shapes
: 
j
NotEqualNotEqualSparseToDense
NotEqual/y*0
_output_shapes
:                  *
T0
c
ToInt32CastNotEqual*

SrcT0
*

DstT0*0
_output_shapes
:                  
`
Sum/reduction_indicesConst*
valueB :
         *
dtype0*
_output_shapes
: 
X
SumSumToInt32Sum/reduction_indices*#
_output_shapes
:         *
T0
o
Reshape/shapeConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
         
m
ReshapeReshapeSparseToDenseReshape/shape"/device:CPU:0*#
_output_shapes
:         *
T0
O
	map/ShapeShapeReshape"/device:CPU:0*
_output_shapes
:*
T0
p
map/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
r
map/strided_slice/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
r
map/strided_slice/stack_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
л
map/strided_sliceStridedSlice	map/Shapemap/strided_slice/stackmap/strided_slice/stack_1map/strided_slice/stack_2"/device:CPU:0*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0
~
map/TensorArrayTensorArrayV3map/strided_slice*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
b
map/TensorArrayUnstack/ShapeShapeReshape"/device:CPU:0*
_output_shapes
:*
T0
Ѓ
*map/TensorArrayUnstack/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
Ё
,map/TensorArrayUnstack/strided_slice/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
Ё
,map/TensorArrayUnstack/strided_slice/stack_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
»
$map/TensorArrayUnstack/strided_sliceStridedSlicemap/TensorArrayUnstack/Shape*map/TensorArrayUnstack/strided_slice/stack,map/TensorArrayUnstack/strided_slice/stack_1,map/TensorArrayUnstack/strided_slice/stack_2"/device:CPU:0*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: 
s
"map/TensorArrayUnstack/range/startConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
s
"map/TensorArrayUnstack/range/deltaConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
К
map/TensorArrayUnstack/rangeRange"map/TensorArrayUnstack/range/start$map/TensorArrayUnstack/strided_slice"map/TensorArrayUnstack/range/delta"/device:CPU:0*#
_output_shapes
:         
ь
>map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map/TensorArraymap/TensorArrayUnstack/rangeReshapemap/TensorArray:1"/device:CPU:0*
_output_shapes
: *
T0*
_class
loc:@Reshape
Z
	map/ConstConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
ђ
map/TensorArray_1TensorArrayV3map/strided_slice*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
l
map/while/iteration_counterConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
ј
map/while/EnterEntermap/while/iteration_counter"/device:CPU:0*
T0*'

frame_namemap/while/while_context*
_output_shapes
: 
~
map/while/Enter_1Enter	map/Const"/device:CPU:0*'

frame_namemap/while/while_context*
_output_shapes
: *
T0
ѕ
map/while/Enter_2Entermap/TensorArray_1:1"/device:CPU:0*
T0*'

frame_namemap/while/while_context*
_output_shapes
: 
}
map/while/MergeMergemap/while/Entermap/while/NextIteration"/device:CPU:0*
_output_shapes
: : *
N*
T0
Ѓ
map/while/Merge_1Mergemap/while/Enter_1map/while/NextIteration_1"/device:CPU:0*
_output_shapes
: : *
N*
T0
Ѓ
map/while/Merge_2Mergemap/while/Enter_2map/while/NextIteration_2"/device:CPU:0*
_output_shapes
: : *
N*
T0
m
map/while/LessLessmap/while/Mergemap/while/Less/Enter"/device:CPU:0*
_output_shapes
: *
T0
ю
map/while/Less/EnterEntermap/strided_slice"/device:CPU:0*
T0*'

frame_namemap/while/while_context*
is_constant(*
_output_shapes
: 
q
map/while/Less_1Lessmap/while/Merge_1map/while/Less/Enter"/device:CPU:0*
_output_shapes
: *
T0
k
map/while/LogicalAnd
LogicalAndmap/while/Lessmap/while/Less_1"/device:CPU:0*
_output_shapes
: 
[
map/while/LoopCondLoopCondmap/while/LogicalAnd"/device:CPU:0*
_output_shapes
: 
Ћ
map/while/SwitchSwitchmap/while/Mergemap/while/LoopCond"/device:CPU:0*"
_class
loc:@map/while/Merge*
_output_shapes
: : *
T0
Џ
map/while/Switch_1Switchmap/while/Merge_1map/while/LoopCond"/device:CPU:0*
T0*$
_class
loc:@map/while/Merge_1*
_output_shapes
: : 
Џ
map/while/Switch_2Switchmap/while/Merge_2map/while/LoopCond"/device:CPU:0*
T0*$
_class
loc:@map/while/Merge_2*
_output_shapes
: : 
b
map/while/IdentityIdentitymap/while/Switch:1"/device:CPU:0*
T0*
_output_shapes
: 
f
map/while/Identity_1Identitymap/while/Switch_1:1"/device:CPU:0*
_output_shapes
: *
T0
f
map/while/Identity_2Identitymap/while/Switch_2:1"/device:CPU:0*
T0*
_output_shapes
: 
u
map/while/add/yConst^map/while/Identity"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
k
map/while/addAddV2map/while/Identitymap/while/add/y"/device:CPU:0*
_output_shapes
: *
T0
┬
map/while/TensorArrayReadV3TensorArrayReadV3!map/while/TensorArrayReadV3/Entermap/while/Identity_1#map/while/TensorArrayReadV3/Enter_1"/device:CPU:0*
dtype0*
_output_shapes
: 
Ф
!map/while/TensorArrayReadV3/EnterEntermap/TensorArray"/device:CPU:0*
T0*'

frame_namemap/while/while_context*
is_constant(*
_output_shapes
:
п
#map/while/TensorArrayReadV3/Enter_1Enter>map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3"/device:CPU:0*
T0*'

frame_namemap/while/while_context*
is_constant(*
_output_shapes
: 
Ђ
map/while/DecodeRaw	DecodeRawmap/while/TensorArrayReadV3"/device:CPU:0*#
_output_shapes
:         *
out_type0
z
map/while/ToInt32Castmap/while/DecodeRaw"/device:CPU:0*

SrcT0*

DstT0*#
_output_shapes
:         
І
map/while/strided_slice/stackConst^map/while/Identity"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
Ї
map/while/strided_slice/stack_1Const^map/while/Identity"/device:CPU:0*
valueB:0*
dtype0*
_output_shapes
:
Ї
map/while/strided_slice/stack_2Const^map/while/Identity"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
э
map/while/strided_sliceStridedSlicemap/while/ToInt32map/while/strided_slice/stackmap/while/strided_slice/stack_1map/while/strided_slice/stack_2"/device:CPU:0*

begin_mask*#
_output_shapes
:         *
Index0*
T0
e
map/while/ShapeShapemap/while/strided_slice"/device:CPU:0*
_output_shapes
:*
T0
Ї
map/while/strided_slice_1/stackConst^map/while/Identity"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
Ј
!map/while/strided_slice_1/stack_1Const^map/while/Identity"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ј
!map/while/strided_slice_1/stack_2Const^map/while/Identity"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ш
map/while/strided_slice_1StridedSlicemap/while/Shapemap/while/strided_slice_1/stack!map/while/strided_slice_1/stack_1!map/while/strided_slice_1/stack_2"/device:CPU:0*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0
u
map/while/sub/xConst^map/while/Identity"/device:CPU:0*
value	B :2*
dtype0*
_output_shapes
: 
p
map/while/subSubmap/while/sub/xmap/while/strided_slice_1"/device:CPU:0*
T0*
_output_shapes
: 
w
map/while/sub_1/yConst^map/while/Identity"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
h
map/while/sub_1Submap/while/submap/while/sub_1/y"/device:CPU:0*
_output_shapes
: *
T0
i
map/while/Fill/dimsPackmap/while/sub_1"/device:CPU:0*
N*
T0*
_output_shapes
:
{
map/while/Fill/valueConst^map/while/Identity"/device:CPU:0*
value
B :ё*
dtype0*
_output_shapes
: 
~
map/while/FillFillmap/while/Fill/dimsmap/while/Fill/value"/device:CPU:0*
T0*#
_output_shapes
:         
ѕ
map/while/concat/values_0Const^map/while/Identity"/device:CPU:0*
valueB:ѓ*
dtype0*
_output_shapes
:
ѕ
map/while/concat/values_2Const^map/while/Identity"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:Ѓ
{
map/while/concat/axisConst^map/while/Identity"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
¤
map/while/concatConcatV2map/while/concat/values_0map/while/strided_slicemap/while/concat/values_2map/while/Fillmap/while/concat/axis"/device:CPU:0*
N*
T0*
_output_shapes
:2
І
-map/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33map/while/TensorArrayWrite/TensorArrayWriteV3/Entermap/while/Identity_1map/while/concatmap/while/Identity_2"/device:CPU:0*#
_class
loc:@map/while/concat*
_output_shapes
: *
T0
С
3map/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap/TensorArray_1"/device:CPU:0*
T0*#
_class
loc:@map/while/concat*'

frame_namemap/while/while_context*
_output_shapes
:*
is_constant(
w
map/while/add_1/yConst^map/while/Identity"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B :
q
map/while/add_1AddV2map/while/Identity_1map/while/add_1/y"/device:CPU:0*
T0*
_output_shapes
: 
g
map/while/NextIterationNextIterationmap/while/add"/device:CPU:0*
_output_shapes
: *
T0
k
map/while/NextIteration_1NextIterationmap/while/add_1"/device:CPU:0*
T0*
_output_shapes
: 
Ѕ
map/while/NextIteration_2NextIteration-map/while/TensorArrayWrite/TensorArrayWriteV3"/device:CPU:0*
_output_shapes
: *
T0
X
map/while/ExitExitmap/while/Switch"/device:CPU:0*
T0*
_output_shapes
: 
\
map/while/Exit_1Exitmap/while/Switch_1"/device:CPU:0*
T0*
_output_shapes
: 
\
map/while/Exit_2Exitmap/while/Switch_2"/device:CPU:0*
T0*
_output_shapes
: 
ѕ
 map/TensorArrayStack/range/startConst*
value	B : *$
_class
loc:@map/TensorArray_1*
dtype0*
_output_shapes
: 
ѕ
 map/TensorArrayStack/range/deltaConst*
value	B :*$
_class
loc:@map/TensorArray_1*
dtype0*
_output_shapes
: 
┼
map/TensorArrayStack/rangeRange map/TensorArrayStack/range/startmap/strided_slice map/TensorArrayStack/range/delta*$
_class
loc:@map/TensorArray_1*#
_output_shapes
:         
ш
(map/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map/TensorArray_1map/TensorArrayStack/rangemap/while/Exit_2*
element_shape:2*$
_class
loc:@map/TensorArray_1*
dtype0*'
_output_shapes
:         2
B
ShapeShapeSparseToDense*
T0*
_output_shapes
:
]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: 
D
Shape_1ShapeSparseToDense*
_output_shapes
:*
T0
_
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
и
strided_slice_1StridedSliceShape_1strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0
S
Reshape_1/shape/2Const*
dtype0*
_output_shapes
: *
value	B :2
x
Reshape_1/shapePackstrided_slicestrided_slice_1Reshape_1/shape/2*
N*
T0*
_output_shapes
:
ј
	Reshape_1Reshape(map/TensorArrayStack/TensorArrayGatherV3Reshape_1/shape*
T0*4
_output_shapes"
 :                  2
Ц
0bilm/char_embed/Initializer/random_uniform/shapeConst*
valueB"     *"
_class
loc:@bilm/char_embed*
dtype0*
_output_shapes
:
Ќ
.bilm/char_embed/Initializer/random_uniform/minConst*
valueB
 *2хЙ*"
_class
loc:@bilm/char_embed*
dtype0*
_output_shapes
: 
Ќ
.bilm/char_embed/Initializer/random_uniform/maxConst*
valueB
 *2х>*"
_class
loc:@bilm/char_embed*
dtype0*
_output_shapes
: 
о
8bilm/char_embed/Initializer/random_uniform/RandomUniformRandomUniform0bilm/char_embed/Initializer/random_uniform/shape*
T0*"
_class
loc:@bilm/char_embed*
dtype0*
_output_shapes
:	Ё
┌
.bilm/char_embed/Initializer/random_uniform/subSub.bilm/char_embed/Initializer/random_uniform/max.bilm/char_embed/Initializer/random_uniform/min*
T0*"
_class
loc:@bilm/char_embed*
_output_shapes
: 
ь
.bilm/char_embed/Initializer/random_uniform/mulMul8bilm/char_embed/Initializer/random_uniform/RandomUniform.bilm/char_embed/Initializer/random_uniform/sub*
T0*"
_class
loc:@bilm/char_embed*
_output_shapes
:	Ё
▀
*bilm/char_embed/Initializer/random_uniformAdd.bilm/char_embed/Initializer/random_uniform/mul.bilm/char_embed/Initializer/random_uniform/min*
T0*"
_class
loc:@bilm/char_embed*
_output_shapes
:	Ё
Ъ
bilm/char_embedVarHandleOp*
dtype0* 
shared_namebilm/char_embed*
_output_shapes
: *
shape:	Ё*"
_class
loc:@bilm/char_embed
o
0bilm/char_embed/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/char_embed*
_output_shapes
: 
t
bilm/char_embed/AssignAssignVariableOpbilm/char_embed*bilm/char_embed/Initializer/random_uniform*
dtype0
t
#bilm/char_embed/Read/ReadVariableOpReadVariableOpbilm/char_embed*
dtype0*
_output_shapes
:	Ё
Й
bilm/embedding_lookupResourceGatherbilm/char_embed	Reshape_1*
dtype0*
Tindices0*8
_output_shapes&
$:"                  2*"
_class
loc:@bilm/char_embed
е
bilm/embedding_lookup/IdentityIdentitybilm/embedding_lookup*8
_output_shapes&
$:"                  2*
T0*"
_class
loc:@bilm/char_embed
Ј
 bilm/embedding_lookup/Identity_1Identitybilm/embedding_lookup/Identity*
T0*8
_output_shapes&
$:"                  2
»
1bilm/CNN/W_cnn_0/Initializer/random_uniform/shapeConst*%
valueB"             *#
_class
loc:@bilm/CNN/W_cnn_0*
dtype0*
_output_shapes
:
Ў
/bilm/CNN/W_cnn_0/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *зхЙ*#
_class
loc:@bilm/CNN/W_cnn_0
Ў
/bilm/CNN/W_cnn_0/Initializer/random_uniform/maxConst*
valueB
 *зх>*#
_class
loc:@bilm/CNN/W_cnn_0*
dtype0*
_output_shapes
: 
Я
9bilm/CNN/W_cnn_0/Initializer/random_uniform/RandomUniformRandomUniform1bilm/CNN/W_cnn_0/Initializer/random_uniform/shape*#
_class
loc:@bilm/CNN/W_cnn_0*
dtype0*&
_output_shapes
: *
T0
я
/bilm/CNN/W_cnn_0/Initializer/random_uniform/subSub/bilm/CNN/W_cnn_0/Initializer/random_uniform/max/bilm/CNN/W_cnn_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@bilm/CNN/W_cnn_0
Э
/bilm/CNN/W_cnn_0/Initializer/random_uniform/mulMul9bilm/CNN/W_cnn_0/Initializer/random_uniform/RandomUniform/bilm/CNN/W_cnn_0/Initializer/random_uniform/sub*
T0*#
_class
loc:@bilm/CNN/W_cnn_0*&
_output_shapes
: 
Ж
+bilm/CNN/W_cnn_0/Initializer/random_uniformAdd/bilm/CNN/W_cnn_0/Initializer/random_uniform/mul/bilm/CNN/W_cnn_0/Initializer/random_uniform/min*
T0*#
_class
loc:@bilm/CNN/W_cnn_0*&
_output_shapes
: 
Е
bilm/CNN/W_cnn_0VarHandleOp*
shape: *#
_class
loc:@bilm/CNN/W_cnn_0*
dtype0*
_output_shapes
: *!
shared_namebilm/CNN/W_cnn_0
q
1bilm/CNN/W_cnn_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN/W_cnn_0*
_output_shapes
: 
w
bilm/CNN/W_cnn_0/AssignAssignVariableOpbilm/CNN/W_cnn_0+bilm/CNN/W_cnn_0/Initializer/random_uniform*
dtype0
}
$bilm/CNN/W_cnn_0/Read/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_0*
dtype0*&
_output_shapes
: 
а
1bilm/CNN/b_cnn_0/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB: *#
_class
loc:@bilm/CNN/b_cnn_0
Ў
/bilm/CNN/b_cnn_0/Initializer/random_uniform/minConst*
valueB
 *q─юЙ*#
_class
loc:@bilm/CNN/b_cnn_0*
dtype0*
_output_shapes
: 
Ў
/bilm/CNN/b_cnn_0/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *q─ю>*#
_class
loc:@bilm/CNN/b_cnn_0
н
9bilm/CNN/b_cnn_0/Initializer/random_uniform/RandomUniformRandomUniform1bilm/CNN/b_cnn_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes
: *
T0*#
_class
loc:@bilm/CNN/b_cnn_0
я
/bilm/CNN/b_cnn_0/Initializer/random_uniform/subSub/bilm/CNN/b_cnn_0/Initializer/random_uniform/max/bilm/CNN/b_cnn_0/Initializer/random_uniform/min*
T0*#
_class
loc:@bilm/CNN/b_cnn_0*
_output_shapes
: 
В
/bilm/CNN/b_cnn_0/Initializer/random_uniform/mulMul9bilm/CNN/b_cnn_0/Initializer/random_uniform/RandomUniform/bilm/CNN/b_cnn_0/Initializer/random_uniform/sub*
T0*#
_class
loc:@bilm/CNN/b_cnn_0*
_output_shapes
: 
я
+bilm/CNN/b_cnn_0/Initializer/random_uniformAdd/bilm/CNN/b_cnn_0/Initializer/random_uniform/mul/bilm/CNN/b_cnn_0/Initializer/random_uniform/min*
T0*#
_class
loc:@bilm/CNN/b_cnn_0*
_output_shapes
: 
Ю
bilm/CNN/b_cnn_0VarHandleOp*
shape: *#
_class
loc:@bilm/CNN/b_cnn_0*
dtype0*!
shared_namebilm/CNN/b_cnn_0*
_output_shapes
: 
q
1bilm/CNN/b_cnn_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN/b_cnn_0*
_output_shapes
: 
w
bilm/CNN/b_cnn_0/AssignAssignVariableOpbilm/CNN/b_cnn_0+bilm/CNN/b_cnn_0/Initializer/random_uniform*
dtype0
q
$bilm/CNN/b_cnn_0/Read/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_0*
dtype0*
_output_shapes
: 
w
bilm/CNN/Conv2D/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_0*
dtype0*&
_output_shapes
: 
К
bilm/CNN/Conv2DConv2D bilm/embedding_lookup/Identity_1bilm/CNN/Conv2D/ReadVariableOp*
T0*
strides
*
paddingVALID*8
_output_shapes&
$:"                  2 
h
bilm/CNN/add/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_0*
dtype0*
_output_shapes
: 
є
bilm/CNN/addAddV2bilm/CNN/Conv2Dbilm/CNN/add/ReadVariableOp*
T0*8
_output_shapes&
$:"                  2 
Ъ
bilm/CNN/MaxPoolMaxPoolbilm/CNN/add*
strides
*
paddingVALID*
ksize
2*8
_output_shapes&
$:"                   
j
bilm/CNN/ReluRelubilm/CNN/MaxPool*
T0*8
_output_shapes&
$:"                   
ђ
bilm/CNN/SqueezeSqueezebilm/CNN/Relu*
squeeze_dims
*4
_output_shapes"
 :                   *
T0
»
1bilm/CNN/W_cnn_1/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"             *#
_class
loc:@bilm/CNN/W_cnn_1
Ў
/bilm/CNN/W_cnn_1/Initializer/random_uniform/minConst*
valueB
 *  ђЙ*#
_class
loc:@bilm/CNN/W_cnn_1*
dtype0*
_output_shapes
: 
Ў
/bilm/CNN/W_cnn_1/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ>*#
_class
loc:@bilm/CNN/W_cnn_1
Я
9bilm/CNN/W_cnn_1/Initializer/random_uniform/RandomUniformRandomUniform1bilm/CNN/W_cnn_1/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
: *
T0*#
_class
loc:@bilm/CNN/W_cnn_1
я
/bilm/CNN/W_cnn_1/Initializer/random_uniform/subSub/bilm/CNN/W_cnn_1/Initializer/random_uniform/max/bilm/CNN/W_cnn_1/Initializer/random_uniform/min*#
_class
loc:@bilm/CNN/W_cnn_1*
_output_shapes
: *
T0
Э
/bilm/CNN/W_cnn_1/Initializer/random_uniform/mulMul9bilm/CNN/W_cnn_1/Initializer/random_uniform/RandomUniform/bilm/CNN/W_cnn_1/Initializer/random_uniform/sub*
T0*#
_class
loc:@bilm/CNN/W_cnn_1*&
_output_shapes
: 
Ж
+bilm/CNN/W_cnn_1/Initializer/random_uniformAdd/bilm/CNN/W_cnn_1/Initializer/random_uniform/mul/bilm/CNN/W_cnn_1/Initializer/random_uniform/min*#
_class
loc:@bilm/CNN/W_cnn_1*&
_output_shapes
: *
T0
Е
bilm/CNN/W_cnn_1VarHandleOp*
dtype0*
_output_shapes
: *!
shared_namebilm/CNN/W_cnn_1*
shape: *#
_class
loc:@bilm/CNN/W_cnn_1
q
1bilm/CNN/W_cnn_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN/W_cnn_1*
_output_shapes
: 
w
bilm/CNN/W_cnn_1/AssignAssignVariableOpbilm/CNN/W_cnn_1+bilm/CNN/W_cnn_1/Initializer/random_uniform*
dtype0
}
$bilm/CNN/W_cnn_1/Read/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_1*
dtype0*&
_output_shapes
: 
а
1bilm/CNN/b_cnn_1/Initializer/random_uniform/shapeConst*
valueB: *#
_class
loc:@bilm/CNN/b_cnn_1*
dtype0*
_output_shapes
:
Ў
/bilm/CNN/b_cnn_1/Initializer/random_uniform/minConst*
valueB
 *q─юЙ*#
_class
loc:@bilm/CNN/b_cnn_1*
dtype0*
_output_shapes
: 
Ў
/bilm/CNN/b_cnn_1/Initializer/random_uniform/maxConst*
valueB
 *q─ю>*#
_class
loc:@bilm/CNN/b_cnn_1*
dtype0*
_output_shapes
: 
н
9bilm/CNN/b_cnn_1/Initializer/random_uniform/RandomUniformRandomUniform1bilm/CNN/b_cnn_1/Initializer/random_uniform/shape*#
_class
loc:@bilm/CNN/b_cnn_1*
dtype0*
_output_shapes
: *
T0
я
/bilm/CNN/b_cnn_1/Initializer/random_uniform/subSub/bilm/CNN/b_cnn_1/Initializer/random_uniform/max/bilm/CNN/b_cnn_1/Initializer/random_uniform/min*#
_class
loc:@bilm/CNN/b_cnn_1*
_output_shapes
: *
T0
В
/bilm/CNN/b_cnn_1/Initializer/random_uniform/mulMul9bilm/CNN/b_cnn_1/Initializer/random_uniform/RandomUniform/bilm/CNN/b_cnn_1/Initializer/random_uniform/sub*
T0*#
_class
loc:@bilm/CNN/b_cnn_1*
_output_shapes
: 
я
+bilm/CNN/b_cnn_1/Initializer/random_uniformAdd/bilm/CNN/b_cnn_1/Initializer/random_uniform/mul/bilm/CNN/b_cnn_1/Initializer/random_uniform/min*#
_class
loc:@bilm/CNN/b_cnn_1*
_output_shapes
: *
T0
Ю
bilm/CNN/b_cnn_1VarHandleOp*
shape: *#
_class
loc:@bilm/CNN/b_cnn_1*
dtype0*
_output_shapes
: *!
shared_namebilm/CNN/b_cnn_1
q
1bilm/CNN/b_cnn_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN/b_cnn_1*
_output_shapes
: 
w
bilm/CNN/b_cnn_1/AssignAssignVariableOpbilm/CNN/b_cnn_1+bilm/CNN/b_cnn_1/Initializer/random_uniform*
dtype0
q
$bilm/CNN/b_cnn_1/Read/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_1*
dtype0*
_output_shapes
: 
y
 bilm/CNN/Conv2D_1/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_1*
dtype0*&
_output_shapes
: 
╦
bilm/CNN/Conv2D_1Conv2D bilm/embedding_lookup/Identity_1 bilm/CNN/Conv2D_1/ReadVariableOp*
strides
*
T0*
paddingVALID*8
_output_shapes&
$:"                  1 
j
bilm/CNN/add_1/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_1*
dtype0*
_output_shapes
: 
ї
bilm/CNN/add_1AddV2bilm/CNN/Conv2D_1bilm/CNN/add_1/ReadVariableOp*8
_output_shapes&
$:"                  1 *
T0
Б
bilm/CNN/MaxPool_1MaxPoolbilm/CNN/add_1*
strides
*
paddingVALID*
ksize
1*8
_output_shapes&
$:"                   
n
bilm/CNN/Relu_1Relubilm/CNN/MaxPool_1*
T0*8
_output_shapes&
$:"                   
ё
bilm/CNN/Squeeze_1Squeezebilm/CNN/Relu_1*
T0*
squeeze_dims
*4
_output_shapes"
 :                   
»
1bilm/CNN/W_cnn_2/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   *#
_class
loc:@bilm/CNN/W_cnn_2
Ў
/bilm/CNN/W_cnn_2/Initializer/random_uniform/minConst*
valueB
 *ЏУ!Й*#
_class
loc:@bilm/CNN/W_cnn_2*
dtype0*
_output_shapes
: 
Ў
/bilm/CNN/W_cnn_2/Initializer/random_uniform/maxConst*
valueB
 *ЏУ!>*#
_class
loc:@bilm/CNN/W_cnn_2*
dtype0*
_output_shapes
: 
Я
9bilm/CNN/W_cnn_2/Initializer/random_uniform/RandomUniformRandomUniform1bilm/CNN/W_cnn_2/Initializer/random_uniform/shape*
T0*#
_class
loc:@bilm/CNN/W_cnn_2*
dtype0*&
_output_shapes
:@
я
/bilm/CNN/W_cnn_2/Initializer/random_uniform/subSub/bilm/CNN/W_cnn_2/Initializer/random_uniform/max/bilm/CNN/W_cnn_2/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@bilm/CNN/W_cnn_2
Э
/bilm/CNN/W_cnn_2/Initializer/random_uniform/mulMul9bilm/CNN/W_cnn_2/Initializer/random_uniform/RandomUniform/bilm/CNN/W_cnn_2/Initializer/random_uniform/sub*
T0*#
_class
loc:@bilm/CNN/W_cnn_2*&
_output_shapes
:@
Ж
+bilm/CNN/W_cnn_2/Initializer/random_uniformAdd/bilm/CNN/W_cnn_2/Initializer/random_uniform/mul/bilm/CNN/W_cnn_2/Initializer/random_uniform/min*
T0*#
_class
loc:@bilm/CNN/W_cnn_2*&
_output_shapes
:@
Е
bilm/CNN/W_cnn_2VarHandleOp*
shape:@*#
_class
loc:@bilm/CNN/W_cnn_2*
dtype0*
_output_shapes
: *!
shared_namebilm/CNN/W_cnn_2
q
1bilm/CNN/W_cnn_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN/W_cnn_2*
_output_shapes
: 
w
bilm/CNN/W_cnn_2/AssignAssignVariableOpbilm/CNN/W_cnn_2+bilm/CNN/W_cnn_2/Initializer/random_uniform*
dtype0
}
$bilm/CNN/W_cnn_2/Read/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_2*
dtype0*&
_output_shapes
:@
а
1bilm/CNN/b_cnn_2/Initializer/random_uniform/shapeConst*
valueB:@*#
_class
loc:@bilm/CNN/b_cnn_2*
dtype0*
_output_shapes
:
Ў
/bilm/CNN/b_cnn_2/Initializer/random_uniform/minConst*
valueB
 *О│]Й*#
_class
loc:@bilm/CNN/b_cnn_2*
dtype0*
_output_shapes
: 
Ў
/bilm/CNN/b_cnn_2/Initializer/random_uniform/maxConst*
valueB
 *О│]>*#
_class
loc:@bilm/CNN/b_cnn_2*
dtype0*
_output_shapes
: 
н
9bilm/CNN/b_cnn_2/Initializer/random_uniform/RandomUniformRandomUniform1bilm/CNN/b_cnn_2/Initializer/random_uniform/shape*
T0*#
_class
loc:@bilm/CNN/b_cnn_2*
dtype0*
_output_shapes
:@
я
/bilm/CNN/b_cnn_2/Initializer/random_uniform/subSub/bilm/CNN/b_cnn_2/Initializer/random_uniform/max/bilm/CNN/b_cnn_2/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@bilm/CNN/b_cnn_2
В
/bilm/CNN/b_cnn_2/Initializer/random_uniform/mulMul9bilm/CNN/b_cnn_2/Initializer/random_uniform/RandomUniform/bilm/CNN/b_cnn_2/Initializer/random_uniform/sub*
T0*#
_class
loc:@bilm/CNN/b_cnn_2*
_output_shapes
:@
я
+bilm/CNN/b_cnn_2/Initializer/random_uniformAdd/bilm/CNN/b_cnn_2/Initializer/random_uniform/mul/bilm/CNN/b_cnn_2/Initializer/random_uniform/min*#
_class
loc:@bilm/CNN/b_cnn_2*
_output_shapes
:@*
T0
Ю
bilm/CNN/b_cnn_2VarHandleOp*
shape:@*#
_class
loc:@bilm/CNN/b_cnn_2*
dtype0*
_output_shapes
: *!
shared_namebilm/CNN/b_cnn_2
q
1bilm/CNN/b_cnn_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN/b_cnn_2*
_output_shapes
: 
w
bilm/CNN/b_cnn_2/AssignAssignVariableOpbilm/CNN/b_cnn_2+bilm/CNN/b_cnn_2/Initializer/random_uniform*
dtype0
q
$bilm/CNN/b_cnn_2/Read/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_2*
dtype0*
_output_shapes
:@
y
 bilm/CNN/Conv2D_2/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_2*
dtype0*&
_output_shapes
:@
╦
bilm/CNN/Conv2D_2Conv2D bilm/embedding_lookup/Identity_1 bilm/CNN/Conv2D_2/ReadVariableOp*8
_output_shapes&
$:"                  0@*
T0*
strides
*
paddingVALID
j
bilm/CNN/add_2/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_2*
dtype0*
_output_shapes
:@
ї
bilm/CNN/add_2AddV2bilm/CNN/Conv2D_2bilm/CNN/add_2/ReadVariableOp*8
_output_shapes&
$:"                  0@*
T0
Б
bilm/CNN/MaxPool_2MaxPoolbilm/CNN/add_2*
strides
*
paddingVALID*
ksize
0*8
_output_shapes&
$:"                  @
n
bilm/CNN/Relu_2Relubilm/CNN/MaxPool_2*8
_output_shapes&
$:"                  @*
T0
ё
bilm/CNN/Squeeze_2Squeezebilm/CNN/Relu_2*
T0*
squeeze_dims
*4
_output_shapes"
 :                  @
»
1bilm/CNN/W_cnn_3/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"         ђ   *#
_class
loc:@bilm/CNN/W_cnn_3
Ў
/bilm/CNN/W_cnn_3/Initializer/random_uniform/minConst*
valueB
 *ВЛй*#
_class
loc:@bilm/CNN/W_cnn_3*
dtype0*
_output_shapes
: 
Ў
/bilm/CNN/W_cnn_3/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ВЛ=*#
_class
loc:@bilm/CNN/W_cnn_3
р
9bilm/CNN/W_cnn_3/Initializer/random_uniform/RandomUniformRandomUniform1bilm/CNN/W_cnn_3/Initializer/random_uniform/shape*
T0*#
_class
loc:@bilm/CNN/W_cnn_3*
dtype0*'
_output_shapes
:ђ
я
/bilm/CNN/W_cnn_3/Initializer/random_uniform/subSub/bilm/CNN/W_cnn_3/Initializer/random_uniform/max/bilm/CNN/W_cnn_3/Initializer/random_uniform/min*
T0*#
_class
loc:@bilm/CNN/W_cnn_3*
_output_shapes
: 
щ
/bilm/CNN/W_cnn_3/Initializer/random_uniform/mulMul9bilm/CNN/W_cnn_3/Initializer/random_uniform/RandomUniform/bilm/CNN/W_cnn_3/Initializer/random_uniform/sub*
T0*#
_class
loc:@bilm/CNN/W_cnn_3*'
_output_shapes
:ђ
в
+bilm/CNN/W_cnn_3/Initializer/random_uniformAdd/bilm/CNN/W_cnn_3/Initializer/random_uniform/mul/bilm/CNN/W_cnn_3/Initializer/random_uniform/min*#
_class
loc:@bilm/CNN/W_cnn_3*'
_output_shapes
:ђ*
T0
ф
bilm/CNN/W_cnn_3VarHandleOp*
shape:ђ*#
_class
loc:@bilm/CNN/W_cnn_3*
dtype0*
_output_shapes
: *!
shared_namebilm/CNN/W_cnn_3
q
1bilm/CNN/W_cnn_3/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN/W_cnn_3*
_output_shapes
: 
w
bilm/CNN/W_cnn_3/AssignAssignVariableOpbilm/CNN/W_cnn_3+bilm/CNN/W_cnn_3/Initializer/random_uniform*
dtype0
~
$bilm/CNN/W_cnn_3/Read/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_3*
dtype0*'
_output_shapes
:ђ
А
1bilm/CNN/b_cnn_3/Initializer/random_uniform/shapeConst*
valueB:ђ*#
_class
loc:@bilm/CNN/b_cnn_3*
dtype0*
_output_shapes
:
Ў
/bilm/CNN/b_cnn_3/Initializer/random_uniform/minConst*
valueB
 *q─Й*#
_class
loc:@bilm/CNN/b_cnn_3*
dtype0*
_output_shapes
: 
Ў
/bilm/CNN/b_cnn_3/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *q─>*#
_class
loc:@bilm/CNN/b_cnn_3
Н
9bilm/CNN/b_cnn_3/Initializer/random_uniform/RandomUniformRandomUniform1bilm/CNN/b_cnn_3/Initializer/random_uniform/shape*#
_class
loc:@bilm/CNN/b_cnn_3*
dtype0*
_output_shapes	
:ђ*
T0
я
/bilm/CNN/b_cnn_3/Initializer/random_uniform/subSub/bilm/CNN/b_cnn_3/Initializer/random_uniform/max/bilm/CNN/b_cnn_3/Initializer/random_uniform/min*
T0*#
_class
loc:@bilm/CNN/b_cnn_3*
_output_shapes
: 
ь
/bilm/CNN/b_cnn_3/Initializer/random_uniform/mulMul9bilm/CNN/b_cnn_3/Initializer/random_uniform/RandomUniform/bilm/CNN/b_cnn_3/Initializer/random_uniform/sub*#
_class
loc:@bilm/CNN/b_cnn_3*
_output_shapes	
:ђ*
T0
▀
+bilm/CNN/b_cnn_3/Initializer/random_uniformAdd/bilm/CNN/b_cnn_3/Initializer/random_uniform/mul/bilm/CNN/b_cnn_3/Initializer/random_uniform/min*
T0*#
_class
loc:@bilm/CNN/b_cnn_3*
_output_shapes	
:ђ
ъ
bilm/CNN/b_cnn_3VarHandleOp*
shape:ђ*#
_class
loc:@bilm/CNN/b_cnn_3*
dtype0*!
shared_namebilm/CNN/b_cnn_3*
_output_shapes
: 
q
1bilm/CNN/b_cnn_3/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN/b_cnn_3*
_output_shapes
: 
w
bilm/CNN/b_cnn_3/AssignAssignVariableOpbilm/CNN/b_cnn_3+bilm/CNN/b_cnn_3/Initializer/random_uniform*
dtype0
r
$bilm/CNN/b_cnn_3/Read/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_3*
dtype0*
_output_shapes	
:ђ
z
 bilm/CNN/Conv2D_3/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_3*
dtype0*'
_output_shapes
:ђ
╠
bilm/CNN/Conv2D_3Conv2D bilm/embedding_lookup/Identity_1 bilm/CNN/Conv2D_3/ReadVariableOp*9
_output_shapes'
%:#                  /ђ*
T0*
strides
*
paddingVALID
k
bilm/CNN/add_3/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_3*
dtype0*
_output_shapes	
:ђ
Ї
bilm/CNN/add_3AddV2bilm/CNN/Conv2D_3bilm/CNN/add_3/ReadVariableOp*
T0*9
_output_shapes'
%:#                  /ђ
ц
bilm/CNN/MaxPool_3MaxPoolbilm/CNN/add_3*
ksize
/*9
_output_shapes'
%:#                  ђ*
strides
*
paddingVALID
o
bilm/CNN/Relu_3Relubilm/CNN/MaxPool_3*9
_output_shapes'
%:#                  ђ*
T0
Ё
bilm/CNN/Squeeze_3Squeezebilm/CNN/Relu_3*
T0*
squeeze_dims
*5
_output_shapes#
!:                  ђ
»
1bilm/CNN/W_cnn_4/Initializer/random_uniform/shapeConst*%
valueB"            *#
_class
loc:@bilm/CNN/W_cnn_4*
dtype0*
_output_shapes
:
Ў
/bilm/CNN/W_cnn_4/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *╦ѕй*#
_class
loc:@bilm/CNN/W_cnn_4
Ў
/bilm/CNN/W_cnn_4/Initializer/random_uniform/maxConst*
valueB
 *╦ѕ=*#
_class
loc:@bilm/CNN/W_cnn_4*
dtype0*
_output_shapes
: 
р
9bilm/CNN/W_cnn_4/Initializer/random_uniform/RandomUniformRandomUniform1bilm/CNN/W_cnn_4/Initializer/random_uniform/shape*
T0*#
_class
loc:@bilm/CNN/W_cnn_4*
dtype0*'
_output_shapes
:ђ
я
/bilm/CNN/W_cnn_4/Initializer/random_uniform/subSub/bilm/CNN/W_cnn_4/Initializer/random_uniform/max/bilm/CNN/W_cnn_4/Initializer/random_uniform/min*
T0*#
_class
loc:@bilm/CNN/W_cnn_4*
_output_shapes
: 
щ
/bilm/CNN/W_cnn_4/Initializer/random_uniform/mulMul9bilm/CNN/W_cnn_4/Initializer/random_uniform/RandomUniform/bilm/CNN/W_cnn_4/Initializer/random_uniform/sub*'
_output_shapes
:ђ*
T0*#
_class
loc:@bilm/CNN/W_cnn_4
в
+bilm/CNN/W_cnn_4/Initializer/random_uniformAdd/bilm/CNN/W_cnn_4/Initializer/random_uniform/mul/bilm/CNN/W_cnn_4/Initializer/random_uniform/min*
T0*#
_class
loc:@bilm/CNN/W_cnn_4*'
_output_shapes
:ђ
ф
bilm/CNN/W_cnn_4VarHandleOp*
shape:ђ*#
_class
loc:@bilm/CNN/W_cnn_4*
dtype0*
_output_shapes
: *!
shared_namebilm/CNN/W_cnn_4
q
1bilm/CNN/W_cnn_4/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN/W_cnn_4*
_output_shapes
: 
w
bilm/CNN/W_cnn_4/AssignAssignVariableOpbilm/CNN/W_cnn_4+bilm/CNN/W_cnn_4/Initializer/random_uniform*
dtype0
~
$bilm/CNN/W_cnn_4/Read/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_4*
dtype0*'
_output_shapes
:ђ
А
1bilm/CNN/b_cnn_4/Initializer/random_uniform/shapeConst*
valueB:ђ*#
_class
loc:@bilm/CNN/b_cnn_4*
dtype0*
_output_shapes
:
Ў
/bilm/CNN/b_cnn_4/Initializer/random_uniform/minConst*
valueB
 *О│Пй*#
_class
loc:@bilm/CNN/b_cnn_4*
dtype0*
_output_shapes
: 
Ў
/bilm/CNN/b_cnn_4/Initializer/random_uniform/maxConst*
valueB
 *О│П=*#
_class
loc:@bilm/CNN/b_cnn_4*
dtype0*
_output_shapes
: 
Н
9bilm/CNN/b_cnn_4/Initializer/random_uniform/RandomUniformRandomUniform1bilm/CNN/b_cnn_4/Initializer/random_uniform/shape*
T0*#
_class
loc:@bilm/CNN/b_cnn_4*
dtype0*
_output_shapes	
:ђ
я
/bilm/CNN/b_cnn_4/Initializer/random_uniform/subSub/bilm/CNN/b_cnn_4/Initializer/random_uniform/max/bilm/CNN/b_cnn_4/Initializer/random_uniform/min*#
_class
loc:@bilm/CNN/b_cnn_4*
_output_shapes
: *
T0
ь
/bilm/CNN/b_cnn_4/Initializer/random_uniform/mulMul9bilm/CNN/b_cnn_4/Initializer/random_uniform/RandomUniform/bilm/CNN/b_cnn_4/Initializer/random_uniform/sub*
_output_shapes	
:ђ*
T0*#
_class
loc:@bilm/CNN/b_cnn_4
▀
+bilm/CNN/b_cnn_4/Initializer/random_uniformAdd/bilm/CNN/b_cnn_4/Initializer/random_uniform/mul/bilm/CNN/b_cnn_4/Initializer/random_uniform/min*#
_class
loc:@bilm/CNN/b_cnn_4*
_output_shapes	
:ђ*
T0
ъ
bilm/CNN/b_cnn_4VarHandleOp*
shape:ђ*#
_class
loc:@bilm/CNN/b_cnn_4*
dtype0*!
shared_namebilm/CNN/b_cnn_4*
_output_shapes
: 
q
1bilm/CNN/b_cnn_4/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN/b_cnn_4*
_output_shapes
: 
w
bilm/CNN/b_cnn_4/AssignAssignVariableOpbilm/CNN/b_cnn_4+bilm/CNN/b_cnn_4/Initializer/random_uniform*
dtype0
r
$bilm/CNN/b_cnn_4/Read/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_4*
dtype0*
_output_shapes	
:ђ
z
 bilm/CNN/Conv2D_4/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_4*
dtype0*'
_output_shapes
:ђ
╠
bilm/CNN/Conv2D_4Conv2D bilm/embedding_lookup/Identity_1 bilm/CNN/Conv2D_4/ReadVariableOp*9
_output_shapes'
%:#                  .ђ*
T0*
strides
*
paddingVALID
k
bilm/CNN/add_4/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_4*
dtype0*
_output_shapes	
:ђ
Ї
bilm/CNN/add_4AddV2bilm/CNN/Conv2D_4bilm/CNN/add_4/ReadVariableOp*9
_output_shapes'
%:#                  .ђ*
T0
ц
bilm/CNN/MaxPool_4MaxPoolbilm/CNN/add_4*
strides
*
paddingVALID*
ksize
.*9
_output_shapes'
%:#                  ђ
o
bilm/CNN/Relu_4Relubilm/CNN/MaxPool_4*
T0*9
_output_shapes'
%:#                  ђ
Ё
bilm/CNN/Squeeze_4Squeezebilm/CNN/Relu_4*
squeeze_dims
*5
_output_shapes#
!:                  ђ*
T0
»
1bilm/CNN/W_cnn_5/Initializer/random_uniform/shapeConst*%
valueB"            *#
_class
loc:@bilm/CNN/W_cnn_5*
dtype0*
_output_shapes
:
Ў
/bilm/CNN/W_cnn_5/Initializer/random_uniform/minConst*
valueB
 *jA2й*#
_class
loc:@bilm/CNN/W_cnn_5*
dtype0*
_output_shapes
: 
Ў
/bilm/CNN/W_cnn_5/Initializer/random_uniform/maxConst*
valueB
 *jA2=*#
_class
loc:@bilm/CNN/W_cnn_5*
dtype0*
_output_shapes
: 
р
9bilm/CNN/W_cnn_5/Initializer/random_uniform/RandomUniformRandomUniform1bilm/CNN/W_cnn_5/Initializer/random_uniform/shape*
T0*#
_class
loc:@bilm/CNN/W_cnn_5*
dtype0*'
_output_shapes
:ђ
я
/bilm/CNN/W_cnn_5/Initializer/random_uniform/subSub/bilm/CNN/W_cnn_5/Initializer/random_uniform/max/bilm/CNN/W_cnn_5/Initializer/random_uniform/min*
T0*#
_class
loc:@bilm/CNN/W_cnn_5*
_output_shapes
: 
щ
/bilm/CNN/W_cnn_5/Initializer/random_uniform/mulMul9bilm/CNN/W_cnn_5/Initializer/random_uniform/RandomUniform/bilm/CNN/W_cnn_5/Initializer/random_uniform/sub*'
_output_shapes
:ђ*
T0*#
_class
loc:@bilm/CNN/W_cnn_5
в
+bilm/CNN/W_cnn_5/Initializer/random_uniformAdd/bilm/CNN/W_cnn_5/Initializer/random_uniform/mul/bilm/CNN/W_cnn_5/Initializer/random_uniform/min*'
_output_shapes
:ђ*
T0*#
_class
loc:@bilm/CNN/W_cnn_5
ф
bilm/CNN/W_cnn_5VarHandleOp*
dtype0*
_output_shapes
: *!
shared_namebilm/CNN/W_cnn_5*
shape:ђ*#
_class
loc:@bilm/CNN/W_cnn_5
q
1bilm/CNN/W_cnn_5/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN/W_cnn_5*
_output_shapes
: 
w
bilm/CNN/W_cnn_5/AssignAssignVariableOpbilm/CNN/W_cnn_5+bilm/CNN/W_cnn_5/Initializer/random_uniform*
dtype0
~
$bilm/CNN/W_cnn_5/Read/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_5*
dtype0*'
_output_shapes
:ђ
А
1bilm/CNN/b_cnn_5/Initializer/random_uniform/shapeConst*
valueB:ђ*#
_class
loc:@bilm/CNN/b_cnn_5*
dtype0*
_output_shapes
:
Ў
/bilm/CNN/b_cnn_5/Initializer/random_uniform/minConst*
valueB
 *q─юй*#
_class
loc:@bilm/CNN/b_cnn_5*
dtype0*
_output_shapes
: 
Ў
/bilm/CNN/b_cnn_5/Initializer/random_uniform/maxConst*
valueB
 *q─ю=*#
_class
loc:@bilm/CNN/b_cnn_5*
dtype0*
_output_shapes
: 
Н
9bilm/CNN/b_cnn_5/Initializer/random_uniform/RandomUniformRandomUniform1bilm/CNN/b_cnn_5/Initializer/random_uniform/shape*#
_class
loc:@bilm/CNN/b_cnn_5*
dtype0*
_output_shapes	
:ђ*
T0
я
/bilm/CNN/b_cnn_5/Initializer/random_uniform/subSub/bilm/CNN/b_cnn_5/Initializer/random_uniform/max/bilm/CNN/b_cnn_5/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@bilm/CNN/b_cnn_5
ь
/bilm/CNN/b_cnn_5/Initializer/random_uniform/mulMul9bilm/CNN/b_cnn_5/Initializer/random_uniform/RandomUniform/bilm/CNN/b_cnn_5/Initializer/random_uniform/sub*
_output_shapes	
:ђ*
T0*#
_class
loc:@bilm/CNN/b_cnn_5
▀
+bilm/CNN/b_cnn_5/Initializer/random_uniformAdd/bilm/CNN/b_cnn_5/Initializer/random_uniform/mul/bilm/CNN/b_cnn_5/Initializer/random_uniform/min*
T0*#
_class
loc:@bilm/CNN/b_cnn_5*
_output_shapes	
:ђ
ъ
bilm/CNN/b_cnn_5VarHandleOp*
shape:ђ*#
_class
loc:@bilm/CNN/b_cnn_5*
dtype0*!
shared_namebilm/CNN/b_cnn_5*
_output_shapes
: 
q
1bilm/CNN/b_cnn_5/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN/b_cnn_5*
_output_shapes
: 
w
bilm/CNN/b_cnn_5/AssignAssignVariableOpbilm/CNN/b_cnn_5+bilm/CNN/b_cnn_5/Initializer/random_uniform*
dtype0
r
$bilm/CNN/b_cnn_5/Read/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_5*
dtype0*
_output_shapes	
:ђ
z
 bilm/CNN/Conv2D_5/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_5*
dtype0*'
_output_shapes
:ђ
╠
bilm/CNN/Conv2D_5Conv2D bilm/embedding_lookup/Identity_1 bilm/CNN/Conv2D_5/ReadVariableOp*9
_output_shapes'
%:#                  -ђ*
T0*
strides
*
paddingVALID
k
bilm/CNN/add_5/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_5*
dtype0*
_output_shapes	
:ђ
Ї
bilm/CNN/add_5AddV2bilm/CNN/Conv2D_5bilm/CNN/add_5/ReadVariableOp*9
_output_shapes'
%:#                  -ђ*
T0
ц
bilm/CNN/MaxPool_5MaxPoolbilm/CNN/add_5*
strides
*
paddingVALID*
ksize
-*9
_output_shapes'
%:#                  ђ
o
bilm/CNN/Relu_5Relubilm/CNN/MaxPool_5*9
_output_shapes'
%:#                  ђ*
T0
Ё
bilm/CNN/Squeeze_5Squeezebilm/CNN/Relu_5*
T0*
squeeze_dims
*5
_output_shapes#
!:                  ђ
»
1bilm/CNN/W_cnn_6/Initializer/random_uniform/shapeConst*%
valueB"            *#
_class
loc:@bilm/CNN/W_cnn_6*
dtype0*
_output_shapes
:
Ў
/bilm/CNN/W_cnn_6/Initializer/random_uniform/minConst*
valueB
 *.в╝*#
_class
loc:@bilm/CNN/W_cnn_6*
dtype0*
_output_shapes
: 
Ў
/bilm/CNN/W_cnn_6/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *.в<*#
_class
loc:@bilm/CNN/W_cnn_6
р
9bilm/CNN/W_cnn_6/Initializer/random_uniform/RandomUniformRandomUniform1bilm/CNN/W_cnn_6/Initializer/random_uniform/shape*#
_class
loc:@bilm/CNN/W_cnn_6*
dtype0*'
_output_shapes
:ђ*
T0
я
/bilm/CNN/W_cnn_6/Initializer/random_uniform/subSub/bilm/CNN/W_cnn_6/Initializer/random_uniform/max/bilm/CNN/W_cnn_6/Initializer/random_uniform/min*#
_class
loc:@bilm/CNN/W_cnn_6*
_output_shapes
: *
T0
щ
/bilm/CNN/W_cnn_6/Initializer/random_uniform/mulMul9bilm/CNN/W_cnn_6/Initializer/random_uniform/RandomUniform/bilm/CNN/W_cnn_6/Initializer/random_uniform/sub*'
_output_shapes
:ђ*
T0*#
_class
loc:@bilm/CNN/W_cnn_6
в
+bilm/CNN/W_cnn_6/Initializer/random_uniformAdd/bilm/CNN/W_cnn_6/Initializer/random_uniform/mul/bilm/CNN/W_cnn_6/Initializer/random_uniform/min*
T0*#
_class
loc:@bilm/CNN/W_cnn_6*'
_output_shapes
:ђ
ф
bilm/CNN/W_cnn_6VarHandleOp*
dtype0*
_output_shapes
: *!
shared_namebilm/CNN/W_cnn_6*
shape:ђ*#
_class
loc:@bilm/CNN/W_cnn_6
q
1bilm/CNN/W_cnn_6/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN/W_cnn_6*
_output_shapes
: 
w
bilm/CNN/W_cnn_6/AssignAssignVariableOpbilm/CNN/W_cnn_6+bilm/CNN/W_cnn_6/Initializer/random_uniform*
dtype0
~
$bilm/CNN/W_cnn_6/Read/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_6*
dtype0*'
_output_shapes
:ђ
А
1bilm/CNN/b_cnn_6/Initializer/random_uniform/shapeConst*
valueB:ђ*#
_class
loc:@bilm/CNN/b_cnn_6*
dtype0*
_output_shapes
:
Ў
/bilm/CNN/b_cnn_6/Initializer/random_uniform/minConst*
valueB
 *О│]й*#
_class
loc:@bilm/CNN/b_cnn_6*
dtype0*
_output_shapes
: 
Ў
/bilm/CNN/b_cnn_6/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *О│]=*#
_class
loc:@bilm/CNN/b_cnn_6
Н
9bilm/CNN/b_cnn_6/Initializer/random_uniform/RandomUniformRandomUniform1bilm/CNN/b_cnn_6/Initializer/random_uniform/shape*
T0*#
_class
loc:@bilm/CNN/b_cnn_6*
dtype0*
_output_shapes	
:ђ
я
/bilm/CNN/b_cnn_6/Initializer/random_uniform/subSub/bilm/CNN/b_cnn_6/Initializer/random_uniform/max/bilm/CNN/b_cnn_6/Initializer/random_uniform/min*
T0*#
_class
loc:@bilm/CNN/b_cnn_6*
_output_shapes
: 
ь
/bilm/CNN/b_cnn_6/Initializer/random_uniform/mulMul9bilm/CNN/b_cnn_6/Initializer/random_uniform/RandomUniform/bilm/CNN/b_cnn_6/Initializer/random_uniform/sub*#
_class
loc:@bilm/CNN/b_cnn_6*
_output_shapes	
:ђ*
T0
▀
+bilm/CNN/b_cnn_6/Initializer/random_uniformAdd/bilm/CNN/b_cnn_6/Initializer/random_uniform/mul/bilm/CNN/b_cnn_6/Initializer/random_uniform/min*
T0*#
_class
loc:@bilm/CNN/b_cnn_6*
_output_shapes	
:ђ
ъ
bilm/CNN/b_cnn_6VarHandleOp*
dtype0*!
shared_namebilm/CNN/b_cnn_6*
_output_shapes
: *
shape:ђ*#
_class
loc:@bilm/CNN/b_cnn_6
q
1bilm/CNN/b_cnn_6/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN/b_cnn_6*
_output_shapes
: 
w
bilm/CNN/b_cnn_6/AssignAssignVariableOpbilm/CNN/b_cnn_6+bilm/CNN/b_cnn_6/Initializer/random_uniform*
dtype0
r
$bilm/CNN/b_cnn_6/Read/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_6*
dtype0*
_output_shapes	
:ђ
z
 bilm/CNN/Conv2D_6/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_6*
dtype0*'
_output_shapes
:ђ
╠
bilm/CNN/Conv2D_6Conv2D bilm/embedding_lookup/Identity_1 bilm/CNN/Conv2D_6/ReadVariableOp*9
_output_shapes'
%:#                  ,ђ*
T0*
strides
*
paddingVALID
k
bilm/CNN/add_6/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_6*
dtype0*
_output_shapes	
:ђ
Ї
bilm/CNN/add_6AddV2bilm/CNN/Conv2D_6bilm/CNN/add_6/ReadVariableOp*
T0*9
_output_shapes'
%:#                  ,ђ
ц
bilm/CNN/MaxPool_6MaxPoolbilm/CNN/add_6*
strides
*
paddingVALID*
ksize
,*9
_output_shapes'
%:#                  ђ
o
bilm/CNN/Relu_6Relubilm/CNN/MaxPool_6*
T0*9
_output_shapes'
%:#                  ђ
Ё
bilm/CNN/Squeeze_6Squeezebilm/CNN/Relu_6*
squeeze_dims
*5
_output_shapes#
!:                  ђ*
T0
R
bilm/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
Ч
bilm/concatConcatV2bilm/CNN/Squeezebilm/CNN/Squeeze_1bilm/CNN/Squeeze_2bilm/CNN/Squeeze_3bilm/CNN/Squeeze_4bilm/CNN/Squeeze_5bilm/CNN/Squeeze_6bilm/concat/axis*5
_output_shapes#
!:                  ђ*
N*
T0
E

bilm/ShapeShapebilm/concat*
_output_shapes
:*
T0
b
bilm/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
d
bilm/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
d
bilm/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
к
bilm/strided_sliceStridedSlice
bilm/Shapebilm/strided_slice/stackbilm/strided_slice/stack_1bilm/strided_slice/stack_2*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0
d
bilm/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
f
bilm/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
f
bilm/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╬
bilm/strided_slice_1StridedSlice
bilm/Shapebilm/strided_slice_1/stackbilm/strided_slice_1/stack_1bilm/strided_slice_1/stack_2*
T0*
_output_shapes
: *
Index0*
shrink_axis_mask
Z
bilm/mulMulbilm/strided_slicebilm/strided_slice_1*
_output_shapes
: *
T0
W
bilm/Reshape/shape/1Const*
value
B :ђ*
dtype0*
_output_shapes
: 
h
bilm/Reshape/shapePackbilm/mulbilm/Reshape/shape/1*
_output_shapes
:*
N*
T0
k
bilm/ReshapeReshapebilm/concatbilm/Reshape/shape*(
_output_shapes
:         ђ*
T0
х
8bilm/CNN_high_0/W_carry/Initializer/random_uniform/shapeConst*
valueB"      **
_class 
loc:@bilm/CNN_high_0/W_carry*
dtype0*
_output_shapes
:
Д
6bilm/CNN_high_0/W_carry/Initializer/random_uniform/minConst*
valueB
 *q─й**
_class 
loc:@bilm/CNN_high_0/W_carry*
dtype0*
_output_shapes
: 
Д
6bilm/CNN_high_0/W_carry/Initializer/random_uniform/maxConst*
valueB
 *q─=**
_class 
loc:@bilm/CNN_high_0/W_carry*
dtype0*
_output_shapes
: 
№
@bilm/CNN_high_0/W_carry/Initializer/random_uniform/RandomUniformRandomUniform8bilm/CNN_high_0/W_carry/Initializer/random_uniform/shape**
_class 
loc:@bilm/CNN_high_0/W_carry*
dtype0* 
_output_shapes
:
ђђ*
T0
Щ
6bilm/CNN_high_0/W_carry/Initializer/random_uniform/subSub6bilm/CNN_high_0/W_carry/Initializer/random_uniform/max6bilm/CNN_high_0/W_carry/Initializer/random_uniform/min*
T0**
_class 
loc:@bilm/CNN_high_0/W_carry*
_output_shapes
: 
ј
6bilm/CNN_high_0/W_carry/Initializer/random_uniform/mulMul@bilm/CNN_high_0/W_carry/Initializer/random_uniform/RandomUniform6bilm/CNN_high_0/W_carry/Initializer/random_uniform/sub* 
_output_shapes
:
ђђ*
T0**
_class 
loc:@bilm/CNN_high_0/W_carry
ђ
2bilm/CNN_high_0/W_carry/Initializer/random_uniformAdd6bilm/CNN_high_0/W_carry/Initializer/random_uniform/mul6bilm/CNN_high_0/W_carry/Initializer/random_uniform/min*
T0**
_class 
loc:@bilm/CNN_high_0/W_carry* 
_output_shapes
:
ђђ
И
bilm/CNN_high_0/W_carryVarHandleOp*
dtype0*(
shared_namebilm/CNN_high_0/W_carry*
_output_shapes
: *
shape:
ђђ**
_class 
loc:@bilm/CNN_high_0/W_carry

8bilm/CNN_high_0/W_carry/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN_high_0/W_carry*
_output_shapes
: 
ї
bilm/CNN_high_0/W_carry/AssignAssignVariableOpbilm/CNN_high_0/W_carry2bilm/CNN_high_0/W_carry/Initializer/random_uniform*
dtype0
Ё
+bilm/CNN_high_0/W_carry/Read/ReadVariableOpReadVariableOpbilm/CNN_high_0/W_carry*
dtype0* 
_output_shapes
:
ђђ
»
8bilm/CNN_high_0/b_carry/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB:ђ**
_class 
loc:@bilm/CNN_high_0/b_carry
Д
6bilm/CNN_high_0/b_carry/Initializer/random_uniform/minConst*
valueB
 *q─й**
_class 
loc:@bilm/CNN_high_0/b_carry*
dtype0*
_output_shapes
: 
Д
6bilm/CNN_high_0/b_carry/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *q─=**
_class 
loc:@bilm/CNN_high_0/b_carry
Ж
@bilm/CNN_high_0/b_carry/Initializer/random_uniform/RandomUniformRandomUniform8bilm/CNN_high_0/b_carry/Initializer/random_uniform/shape*
T0**
_class 
loc:@bilm/CNN_high_0/b_carry*
dtype0*
_output_shapes	
:ђ
Щ
6bilm/CNN_high_0/b_carry/Initializer/random_uniform/subSub6bilm/CNN_high_0/b_carry/Initializer/random_uniform/max6bilm/CNN_high_0/b_carry/Initializer/random_uniform/min*
T0**
_class 
loc:@bilm/CNN_high_0/b_carry*
_output_shapes
: 
Ѕ
6bilm/CNN_high_0/b_carry/Initializer/random_uniform/mulMul@bilm/CNN_high_0/b_carry/Initializer/random_uniform/RandomUniform6bilm/CNN_high_0/b_carry/Initializer/random_uniform/sub**
_class 
loc:@bilm/CNN_high_0/b_carry*
_output_shapes	
:ђ*
T0
ч
2bilm/CNN_high_0/b_carry/Initializer/random_uniformAdd6bilm/CNN_high_0/b_carry/Initializer/random_uniform/mul6bilm/CNN_high_0/b_carry/Initializer/random_uniform/min*
_output_shapes	
:ђ*
T0**
_class 
loc:@bilm/CNN_high_0/b_carry
│
bilm/CNN_high_0/b_carryVarHandleOp*
shape:ђ**
_class 
loc:@bilm/CNN_high_0/b_carry*
dtype0*
_output_shapes
: *(
shared_namebilm/CNN_high_0/b_carry

8bilm/CNN_high_0/b_carry/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN_high_0/b_carry*
_output_shapes
: 
ї
bilm/CNN_high_0/b_carry/AssignAssignVariableOpbilm/CNN_high_0/b_carry2bilm/CNN_high_0/b_carry/Initializer/random_uniform*
dtype0
ђ
+bilm/CNN_high_0/b_carry/Read/ReadVariableOpReadVariableOpbilm/CNN_high_0/b_carry*
dtype0*
_output_shapes	
:ђ
й
<bilm/CNN_high_0/W_transform/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *.
_class$
" loc:@bilm/CNN_high_0/W_transform
»
:bilm/CNN_high_0/W_transform/Initializer/random_uniform/minConst*
valueB
 *q─й*.
_class$
" loc:@bilm/CNN_high_0/W_transform*
dtype0*
_output_shapes
: 
»
:bilm/CNN_high_0/W_transform/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *q─=*.
_class$
" loc:@bilm/CNN_high_0/W_transform
ч
Dbilm/CNN_high_0/W_transform/Initializer/random_uniform/RandomUniformRandomUniform<bilm/CNN_high_0/W_transform/Initializer/random_uniform/shape*.
_class$
" loc:@bilm/CNN_high_0/W_transform*
dtype0* 
_output_shapes
:
ђђ*
T0
і
:bilm/CNN_high_0/W_transform/Initializer/random_uniform/subSub:bilm/CNN_high_0/W_transform/Initializer/random_uniform/max:bilm/CNN_high_0/W_transform/Initializer/random_uniform/min*
_output_shapes
: *
T0*.
_class$
" loc:@bilm/CNN_high_0/W_transform
ъ
:bilm/CNN_high_0/W_transform/Initializer/random_uniform/mulMulDbilm/CNN_high_0/W_transform/Initializer/random_uniform/RandomUniform:bilm/CNN_high_0/W_transform/Initializer/random_uniform/sub*.
_class$
" loc:@bilm/CNN_high_0/W_transform* 
_output_shapes
:
ђђ*
T0
љ
6bilm/CNN_high_0/W_transform/Initializer/random_uniformAdd:bilm/CNN_high_0/W_transform/Initializer/random_uniform/mul:bilm/CNN_high_0/W_transform/Initializer/random_uniform/min*
T0*.
_class$
" loc:@bilm/CNN_high_0/W_transform* 
_output_shapes
:
ђђ
─
bilm/CNN_high_0/W_transformVarHandleOp*
shape:
ђђ*.
_class$
" loc:@bilm/CNN_high_0/W_transform*
dtype0*
_output_shapes
: *,
shared_namebilm/CNN_high_0/W_transform
Є
<bilm/CNN_high_0/W_transform/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN_high_0/W_transform*
_output_shapes
: 
ў
"bilm/CNN_high_0/W_transform/AssignAssignVariableOpbilm/CNN_high_0/W_transform6bilm/CNN_high_0/W_transform/Initializer/random_uniform*
dtype0
Ї
/bilm/CNN_high_0/W_transform/Read/ReadVariableOpReadVariableOpbilm/CNN_high_0/W_transform*
dtype0* 
_output_shapes
:
ђђ
и
<bilm/CNN_high_0/b_transform/Initializer/random_uniform/shapeConst*
valueB:ђ*.
_class$
" loc:@bilm/CNN_high_0/b_transform*
dtype0*
_output_shapes
:
»
:bilm/CNN_high_0/b_transform/Initializer/random_uniform/minConst*
valueB
 *q─й*.
_class$
" loc:@bilm/CNN_high_0/b_transform*
dtype0*
_output_shapes
: 
»
:bilm/CNN_high_0/b_transform/Initializer/random_uniform/maxConst*
valueB
 *q─=*.
_class$
" loc:@bilm/CNN_high_0/b_transform*
dtype0*
_output_shapes
: 
Ш
Dbilm/CNN_high_0/b_transform/Initializer/random_uniform/RandomUniformRandomUniform<bilm/CNN_high_0/b_transform/Initializer/random_uniform/shape*
T0*.
_class$
" loc:@bilm/CNN_high_0/b_transform*
dtype0*
_output_shapes	
:ђ
і
:bilm/CNN_high_0/b_transform/Initializer/random_uniform/subSub:bilm/CNN_high_0/b_transform/Initializer/random_uniform/max:bilm/CNN_high_0/b_transform/Initializer/random_uniform/min*
_output_shapes
: *
T0*.
_class$
" loc:@bilm/CNN_high_0/b_transform
Ў
:bilm/CNN_high_0/b_transform/Initializer/random_uniform/mulMulDbilm/CNN_high_0/b_transform/Initializer/random_uniform/RandomUniform:bilm/CNN_high_0/b_transform/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@bilm/CNN_high_0/b_transform*
_output_shapes	
:ђ
І
6bilm/CNN_high_0/b_transform/Initializer/random_uniformAdd:bilm/CNN_high_0/b_transform/Initializer/random_uniform/mul:bilm/CNN_high_0/b_transform/Initializer/random_uniform/min*
T0*.
_class$
" loc:@bilm/CNN_high_0/b_transform*
_output_shapes	
:ђ
┐
bilm/CNN_high_0/b_transformVarHandleOp*
shape:ђ*.
_class$
" loc:@bilm/CNN_high_0/b_transform*
dtype0*,
shared_namebilm/CNN_high_0/b_transform*
_output_shapes
: 
Є
<bilm/CNN_high_0/b_transform/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN_high_0/b_transform*
_output_shapes
: 
ў
"bilm/CNN_high_0/b_transform/AssignAssignVariableOpbilm/CNN_high_0/b_transform6bilm/CNN_high_0/b_transform/Initializer/random_uniform*
dtype0
ѕ
/bilm/CNN_high_0/b_transform/Read/ReadVariableOpReadVariableOpbilm/CNN_high_0/b_transform*
dtype0*
_output_shapes	
:ђ
t
bilm/MatMul/ReadVariableOpReadVariableOpbilm/CNN_high_0/W_carry*
dtype0* 
_output_shapes
:
ђђ
r
bilm/MatMulMatMulbilm/Reshapebilm/MatMul/ReadVariableOp*(
_output_shapes
:         ђ*
T0
l
bilm/add/ReadVariableOpReadVariableOpbilm/CNN_high_0/b_carry*
dtype0*
_output_shapes	
:ђ
j
bilm/addAddV2bilm/MatMulbilm/add/ReadVariableOp*(
_output_shapes
:         ђ*
T0
T
bilm/SigmoidSigmoidbilm/add*
T0*(
_output_shapes
:         ђ
z
bilm/MatMul_1/ReadVariableOpReadVariableOpbilm/CNN_high_0/W_transform*
dtype0* 
_output_shapes
:
ђђ
v
bilm/MatMul_1MatMulbilm/Reshapebilm/MatMul_1/ReadVariableOp*(
_output_shapes
:         ђ*
T0
r
bilm/add_1/ReadVariableOpReadVariableOpbilm/CNN_high_0/b_transform*
dtype0*
_output_shapes	
:ђ
p

bilm/add_1AddV2bilm/MatMul_1bilm/add_1/ReadVariableOp*
T0*(
_output_shapes
:         ђ
P
	bilm/ReluRelu
bilm/add_1*
T0*(
_output_shapes
:         ђ
]

bilm/mul_1Mulbilm/Sigmoid	bilm/Relu*(
_output_shapes
:         ђ*
T0
O

bilm/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
\
bilm/subSub
bilm/sub/xbilm/Sigmoid*(
_output_shapes
:         ђ*
T0
\

bilm/mul_2Mulbilm/subbilm/Reshape*(
_output_shapes
:         ђ*
T0
^

bilm/add_2AddV2
bilm/mul_1
bilm/mul_2*
T0*(
_output_shapes
:         ђ
х
8bilm/CNN_high_1/W_carry/Initializer/random_uniform/shapeConst*
valueB"      **
_class 
loc:@bilm/CNN_high_1/W_carry*
dtype0*
_output_shapes
:
Д
6bilm/CNN_high_1/W_carry/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *q─й**
_class 
loc:@bilm/CNN_high_1/W_carry
Д
6bilm/CNN_high_1/W_carry/Initializer/random_uniform/maxConst*
valueB
 *q─=**
_class 
loc:@bilm/CNN_high_1/W_carry*
dtype0*
_output_shapes
: 
№
@bilm/CNN_high_1/W_carry/Initializer/random_uniform/RandomUniformRandomUniform8bilm/CNN_high_1/W_carry/Initializer/random_uniform/shape*
T0**
_class 
loc:@bilm/CNN_high_1/W_carry*
dtype0* 
_output_shapes
:
ђђ
Щ
6bilm/CNN_high_1/W_carry/Initializer/random_uniform/subSub6bilm/CNN_high_1/W_carry/Initializer/random_uniform/max6bilm/CNN_high_1/W_carry/Initializer/random_uniform/min*
T0**
_class 
loc:@bilm/CNN_high_1/W_carry*
_output_shapes
: 
ј
6bilm/CNN_high_1/W_carry/Initializer/random_uniform/mulMul@bilm/CNN_high_1/W_carry/Initializer/random_uniform/RandomUniform6bilm/CNN_high_1/W_carry/Initializer/random_uniform/sub**
_class 
loc:@bilm/CNN_high_1/W_carry* 
_output_shapes
:
ђђ*
T0
ђ
2bilm/CNN_high_1/W_carry/Initializer/random_uniformAdd6bilm/CNN_high_1/W_carry/Initializer/random_uniform/mul6bilm/CNN_high_1/W_carry/Initializer/random_uniform/min*
T0**
_class 
loc:@bilm/CNN_high_1/W_carry* 
_output_shapes
:
ђђ
И
bilm/CNN_high_1/W_carryVarHandleOp*
shape:
ђђ**
_class 
loc:@bilm/CNN_high_1/W_carry*
dtype0*
_output_shapes
: *(
shared_namebilm/CNN_high_1/W_carry

8bilm/CNN_high_1/W_carry/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN_high_1/W_carry*
_output_shapes
: 
ї
bilm/CNN_high_1/W_carry/AssignAssignVariableOpbilm/CNN_high_1/W_carry2bilm/CNN_high_1/W_carry/Initializer/random_uniform*
dtype0
Ё
+bilm/CNN_high_1/W_carry/Read/ReadVariableOpReadVariableOpbilm/CNN_high_1/W_carry*
dtype0* 
_output_shapes
:
ђђ
»
8bilm/CNN_high_1/b_carry/Initializer/random_uniform/shapeConst*
valueB:ђ**
_class 
loc:@bilm/CNN_high_1/b_carry*
dtype0*
_output_shapes
:
Д
6bilm/CNN_high_1/b_carry/Initializer/random_uniform/minConst*
valueB
 *q─й**
_class 
loc:@bilm/CNN_high_1/b_carry*
dtype0*
_output_shapes
: 
Д
6bilm/CNN_high_1/b_carry/Initializer/random_uniform/maxConst*
valueB
 *q─=**
_class 
loc:@bilm/CNN_high_1/b_carry*
dtype0*
_output_shapes
: 
Ж
@bilm/CNN_high_1/b_carry/Initializer/random_uniform/RandomUniformRandomUniform8bilm/CNN_high_1/b_carry/Initializer/random_uniform/shape*
T0**
_class 
loc:@bilm/CNN_high_1/b_carry*
dtype0*
_output_shapes	
:ђ
Щ
6bilm/CNN_high_1/b_carry/Initializer/random_uniform/subSub6bilm/CNN_high_1/b_carry/Initializer/random_uniform/max6bilm/CNN_high_1/b_carry/Initializer/random_uniform/min*
_output_shapes
: *
T0**
_class 
loc:@bilm/CNN_high_1/b_carry
Ѕ
6bilm/CNN_high_1/b_carry/Initializer/random_uniform/mulMul@bilm/CNN_high_1/b_carry/Initializer/random_uniform/RandomUniform6bilm/CNN_high_1/b_carry/Initializer/random_uniform/sub**
_class 
loc:@bilm/CNN_high_1/b_carry*
_output_shapes	
:ђ*
T0
ч
2bilm/CNN_high_1/b_carry/Initializer/random_uniformAdd6bilm/CNN_high_1/b_carry/Initializer/random_uniform/mul6bilm/CNN_high_1/b_carry/Initializer/random_uniform/min*
T0**
_class 
loc:@bilm/CNN_high_1/b_carry*
_output_shapes	
:ђ
│
bilm/CNN_high_1/b_carryVarHandleOp*
shape:ђ**
_class 
loc:@bilm/CNN_high_1/b_carry*
dtype0*(
shared_namebilm/CNN_high_1/b_carry*
_output_shapes
: 

8bilm/CNN_high_1/b_carry/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN_high_1/b_carry*
_output_shapes
: 
ї
bilm/CNN_high_1/b_carry/AssignAssignVariableOpbilm/CNN_high_1/b_carry2bilm/CNN_high_1/b_carry/Initializer/random_uniform*
dtype0
ђ
+bilm/CNN_high_1/b_carry/Read/ReadVariableOpReadVariableOpbilm/CNN_high_1/b_carry*
dtype0*
_output_shapes	
:ђ
й
<bilm/CNN_high_1/W_transform/Initializer/random_uniform/shapeConst*
valueB"      *.
_class$
" loc:@bilm/CNN_high_1/W_transform*
dtype0*
_output_shapes
:
»
:bilm/CNN_high_1/W_transform/Initializer/random_uniform/minConst*
valueB
 *q─й*.
_class$
" loc:@bilm/CNN_high_1/W_transform*
dtype0*
_output_shapes
: 
»
:bilm/CNN_high_1/W_transform/Initializer/random_uniform/maxConst*
valueB
 *q─=*.
_class$
" loc:@bilm/CNN_high_1/W_transform*
dtype0*
_output_shapes
: 
ч
Dbilm/CNN_high_1/W_transform/Initializer/random_uniform/RandomUniformRandomUniform<bilm/CNN_high_1/W_transform/Initializer/random_uniform/shape*
T0*.
_class$
" loc:@bilm/CNN_high_1/W_transform*
dtype0* 
_output_shapes
:
ђђ
і
:bilm/CNN_high_1/W_transform/Initializer/random_uniform/subSub:bilm/CNN_high_1/W_transform/Initializer/random_uniform/max:bilm/CNN_high_1/W_transform/Initializer/random_uniform/min*
_output_shapes
: *
T0*.
_class$
" loc:@bilm/CNN_high_1/W_transform
ъ
:bilm/CNN_high_1/W_transform/Initializer/random_uniform/mulMulDbilm/CNN_high_1/W_transform/Initializer/random_uniform/RandomUniform:bilm/CNN_high_1/W_transform/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@bilm/CNN_high_1/W_transform* 
_output_shapes
:
ђђ
љ
6bilm/CNN_high_1/W_transform/Initializer/random_uniformAdd:bilm/CNN_high_1/W_transform/Initializer/random_uniform/mul:bilm/CNN_high_1/W_transform/Initializer/random_uniform/min*
T0*.
_class$
" loc:@bilm/CNN_high_1/W_transform* 
_output_shapes
:
ђђ
─
bilm/CNN_high_1/W_transformVarHandleOp*
dtype0*
_output_shapes
: *,
shared_namebilm/CNN_high_1/W_transform*
shape:
ђђ*.
_class$
" loc:@bilm/CNN_high_1/W_transform
Є
<bilm/CNN_high_1/W_transform/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN_high_1/W_transform*
_output_shapes
: 
ў
"bilm/CNN_high_1/W_transform/AssignAssignVariableOpbilm/CNN_high_1/W_transform6bilm/CNN_high_1/W_transform/Initializer/random_uniform*
dtype0
Ї
/bilm/CNN_high_1/W_transform/Read/ReadVariableOpReadVariableOpbilm/CNN_high_1/W_transform*
dtype0* 
_output_shapes
:
ђђ
и
<bilm/CNN_high_1/b_transform/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB:ђ*.
_class$
" loc:@bilm/CNN_high_1/b_transform
»
:bilm/CNN_high_1/b_transform/Initializer/random_uniform/minConst*
valueB
 *q─й*.
_class$
" loc:@bilm/CNN_high_1/b_transform*
dtype0*
_output_shapes
: 
»
:bilm/CNN_high_1/b_transform/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *q─=*.
_class$
" loc:@bilm/CNN_high_1/b_transform
Ш
Dbilm/CNN_high_1/b_transform/Initializer/random_uniform/RandomUniformRandomUniform<bilm/CNN_high_1/b_transform/Initializer/random_uniform/shape*
T0*.
_class$
" loc:@bilm/CNN_high_1/b_transform*
dtype0*
_output_shapes	
:ђ
і
:bilm/CNN_high_1/b_transform/Initializer/random_uniform/subSub:bilm/CNN_high_1/b_transform/Initializer/random_uniform/max:bilm/CNN_high_1/b_transform/Initializer/random_uniform/min*
T0*.
_class$
" loc:@bilm/CNN_high_1/b_transform*
_output_shapes
: 
Ў
:bilm/CNN_high_1/b_transform/Initializer/random_uniform/mulMulDbilm/CNN_high_1/b_transform/Initializer/random_uniform/RandomUniform:bilm/CNN_high_1/b_transform/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@bilm/CNN_high_1/b_transform*
_output_shapes	
:ђ
І
6bilm/CNN_high_1/b_transform/Initializer/random_uniformAdd:bilm/CNN_high_1/b_transform/Initializer/random_uniform/mul:bilm/CNN_high_1/b_transform/Initializer/random_uniform/min*
T0*.
_class$
" loc:@bilm/CNN_high_1/b_transform*
_output_shapes	
:ђ
┐
bilm/CNN_high_1/b_transformVarHandleOp*
shape:ђ*.
_class$
" loc:@bilm/CNN_high_1/b_transform*
dtype0*,
shared_namebilm/CNN_high_1/b_transform*
_output_shapes
: 
Є
<bilm/CNN_high_1/b_transform/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN_high_1/b_transform*
_output_shapes
: 
ў
"bilm/CNN_high_1/b_transform/AssignAssignVariableOpbilm/CNN_high_1/b_transform6bilm/CNN_high_1/b_transform/Initializer/random_uniform*
dtype0
ѕ
/bilm/CNN_high_1/b_transform/Read/ReadVariableOpReadVariableOpbilm/CNN_high_1/b_transform*
dtype0*
_output_shapes	
:ђ
v
bilm/MatMul_2/ReadVariableOpReadVariableOpbilm/CNN_high_1/W_carry*
dtype0* 
_output_shapes
:
ђђ
t
bilm/MatMul_2MatMul
bilm/add_2bilm/MatMul_2/ReadVariableOp*(
_output_shapes
:         ђ*
T0
n
bilm/add_3/ReadVariableOpReadVariableOpbilm/CNN_high_1/b_carry*
dtype0*
_output_shapes	
:ђ
p

bilm/add_3AddV2bilm/MatMul_2bilm/add_3/ReadVariableOp*
T0*(
_output_shapes
:         ђ
X
bilm/Sigmoid_1Sigmoid
bilm/add_3*(
_output_shapes
:         ђ*
T0
z
bilm/MatMul_3/ReadVariableOpReadVariableOpbilm/CNN_high_1/W_transform*
dtype0* 
_output_shapes
:
ђђ
t
bilm/MatMul_3MatMul
bilm/add_2bilm/MatMul_3/ReadVariableOp*
T0*(
_output_shapes
:         ђ
r
bilm/add_4/ReadVariableOpReadVariableOpbilm/CNN_high_1/b_transform*
dtype0*
_output_shapes	
:ђ
p

bilm/add_4AddV2bilm/MatMul_3bilm/add_4/ReadVariableOp*(
_output_shapes
:         ђ*
T0
R
bilm/Relu_1Relu
bilm/add_4*(
_output_shapes
:         ђ*
T0
a

bilm/mul_3Mulbilm/Sigmoid_1bilm/Relu_1*
T0*(
_output_shapes
:         ђ
Q
bilm/sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
b

bilm/sub_1Subbilm/sub_1/xbilm/Sigmoid_1*(
_output_shapes
:         ђ*
T0
\

bilm/mul_4Mul
bilm/sub_1
bilm/add_2*
T0*(
_output_shapes
:         ђ
^

bilm/add_5AddV2
bilm/mul_3
bilm/mul_4*(
_output_shapes
:         ђ*
T0
»
5bilm/CNN_proj/W_proj/Initializer/random_uniform/shapeConst*
valueB"      *'
_class
loc:@bilm/CNN_proj/W_proj*
dtype0*
_output_shapes
:
А
3bilm/CNN_proj/W_proj/Initializer/random_uniform/minConst*
valueB
 *ЭKFй*'
_class
loc:@bilm/CNN_proj/W_proj*
dtype0*
_output_shapes
: 
А
3bilm/CNN_proj/W_proj/Initializer/random_uniform/maxConst*
valueB
 *ЭKF=*'
_class
loc:@bilm/CNN_proj/W_proj*
dtype0*
_output_shapes
: 
Т
=bilm/CNN_proj/W_proj/Initializer/random_uniform/RandomUniformRandomUniform5bilm/CNN_proj/W_proj/Initializer/random_uniform/shape*
T0*'
_class
loc:@bilm/CNN_proj/W_proj*
dtype0* 
_output_shapes
:
ђђ
Ь
3bilm/CNN_proj/W_proj/Initializer/random_uniform/subSub3bilm/CNN_proj/W_proj/Initializer/random_uniform/max3bilm/CNN_proj/W_proj/Initializer/random_uniform/min*
_output_shapes
: *
T0*'
_class
loc:@bilm/CNN_proj/W_proj
ѓ
3bilm/CNN_proj/W_proj/Initializer/random_uniform/mulMul=bilm/CNN_proj/W_proj/Initializer/random_uniform/RandomUniform3bilm/CNN_proj/W_proj/Initializer/random_uniform/sub*
T0*'
_class
loc:@bilm/CNN_proj/W_proj* 
_output_shapes
:
ђђ
З
/bilm/CNN_proj/W_proj/Initializer/random_uniformAdd3bilm/CNN_proj/W_proj/Initializer/random_uniform/mul3bilm/CNN_proj/W_proj/Initializer/random_uniform/min*'
_class
loc:@bilm/CNN_proj/W_proj* 
_output_shapes
:
ђђ*
T0
»
bilm/CNN_proj/W_projVarHandleOp*
dtype0*%
shared_namebilm/CNN_proj/W_proj*
_output_shapes
: *
shape:
ђђ*'
_class
loc:@bilm/CNN_proj/W_proj
y
5bilm/CNN_proj/W_proj/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN_proj/W_proj*
_output_shapes
: 
Ѓ
bilm/CNN_proj/W_proj/AssignAssignVariableOpbilm/CNN_proj/W_proj/bilm/CNN_proj/W_proj/Initializer/random_uniform*
dtype0

(bilm/CNN_proj/W_proj/Read/ReadVariableOpReadVariableOpbilm/CNN_proj/W_proj*
dtype0* 
_output_shapes
:
ђђ
Е
5bilm/CNN_proj/b_proj/Initializer/random_uniform/shapeConst*
valueB:ђ*'
_class
loc:@bilm/CNN_proj/b_proj*
dtype0*
_output_shapes
:
А
3bilm/CNN_proj/b_proj/Initializer/random_uniform/minConst*
valueB
 *q─юй*'
_class
loc:@bilm/CNN_proj/b_proj*
dtype0*
_output_shapes
: 
А
3bilm/CNN_proj/b_proj/Initializer/random_uniform/maxConst*
valueB
 *q─ю=*'
_class
loc:@bilm/CNN_proj/b_proj*
dtype0*
_output_shapes
: 
р
=bilm/CNN_proj/b_proj/Initializer/random_uniform/RandomUniformRandomUniform5bilm/CNN_proj/b_proj/Initializer/random_uniform/shape*
T0*'
_class
loc:@bilm/CNN_proj/b_proj*
dtype0*
_output_shapes	
:ђ
Ь
3bilm/CNN_proj/b_proj/Initializer/random_uniform/subSub3bilm/CNN_proj/b_proj/Initializer/random_uniform/max3bilm/CNN_proj/b_proj/Initializer/random_uniform/min*
_output_shapes
: *
T0*'
_class
loc:@bilm/CNN_proj/b_proj
§
3bilm/CNN_proj/b_proj/Initializer/random_uniform/mulMul=bilm/CNN_proj/b_proj/Initializer/random_uniform/RandomUniform3bilm/CNN_proj/b_proj/Initializer/random_uniform/sub*
_output_shapes	
:ђ*
T0*'
_class
loc:@bilm/CNN_proj/b_proj
№
/bilm/CNN_proj/b_proj/Initializer/random_uniformAdd3bilm/CNN_proj/b_proj/Initializer/random_uniform/mul3bilm/CNN_proj/b_proj/Initializer/random_uniform/min*
_output_shapes	
:ђ*
T0*'
_class
loc:@bilm/CNN_proj/b_proj
ф
bilm/CNN_proj/b_projVarHandleOp*
dtype0*%
shared_namebilm/CNN_proj/b_proj*
_output_shapes
: *
shape:ђ*'
_class
loc:@bilm/CNN_proj/b_proj
y
5bilm/CNN_proj/b_proj/IsInitialized/VarIsInitializedOpVarIsInitializedOpbilm/CNN_proj/b_proj*
_output_shapes
: 
Ѓ
bilm/CNN_proj/b_proj/AssignAssignVariableOpbilm/CNN_proj/b_proj/bilm/CNN_proj/b_proj/Initializer/random_uniform*
dtype0
z
(bilm/CNN_proj/b_proj/Read/ReadVariableOpReadVariableOpbilm/CNN_proj/b_proj*
dtype0*
_output_shapes	
:ђ
s
bilm/MatMul_4/ReadVariableOpReadVariableOpbilm/CNN_proj/W_proj*
dtype0* 
_output_shapes
:
ђђ
t
bilm/MatMul_4MatMul
bilm/add_5bilm/MatMul_4/ReadVariableOp*(
_output_shapes
:         ђ*
T0
k
bilm/add_6/ReadVariableOpReadVariableOpbilm/CNN_proj/b_proj*
dtype0*
_output_shapes	
:ђ
p

bilm/add_6AddV2bilm/MatMul_4bilm/add_6/ReadVariableOp*
T0*(
_output_shapes
:         ђ
Y
bilm/Reshape_1/shape/2Const*
value
B :ђ*
dtype0*
_output_shapes
: 
ї
bilm/Reshape_1/shapePackbilm/strided_slicebilm/strided_slice_1bilm/Reshape_1/shape/2*
N*
T0*
_output_shapes
:
{
bilm/Reshape_1Reshape
bilm/add_6bilm/Reshape_1/shape*
T0*5
_output_shapes#
!:                  ђ
ф
bilm/ExpandDims/inputConst*Я
valueоBМ2"╚                                                                                                     *
dtype0*
_output_shapes
:2
U
bilm/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
r
bilm/ExpandDims
ExpandDimsbilm/ExpandDims/inputbilm/ExpandDims/dim*
_output_shapes

:2*
T0
W
bilm/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
t
bilm/ExpandDims_1
ExpandDimsbilm/ExpandDimsbilm/ExpandDims_1/dim*
T0*"
_output_shapes
:2
г
bilm/ExpandDims_2/inputConst*
dtype0*
_output_shapes
:2*Я
valueоBМ2"╚                                                                                                    
W
bilm/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
x
bilm/ExpandDims_2
ExpandDimsbilm/ExpandDims_2/inputbilm/ExpandDims_2/dim*
T0*
_output_shapes

:2
W
bilm/ExpandDims_3/dimConst*
dtype0*
_output_shapes
: *
value	B :
v
bilm/ExpandDims_3
ExpandDimsbilm/ExpandDims_2bilm/ExpandDims_3/dim*
T0*"
_output_shapes
:2
Х
bilm/embedding_lookup_1ResourceGatherbilm/char_embedbilm/ExpandDims_1*"
_class
loc:@bilm/char_embed*
Tindices0*
dtype0*&
_output_shapes
:2
џ
 bilm/embedding_lookup_1/IdentityIdentitybilm/embedding_lookup_1*
T0*"
_class
loc:@bilm/char_embed*&
_output_shapes
:2
Ђ
"bilm/embedding_lookup_1/Identity_1Identity bilm/embedding_lookup_1/Identity*
T0*&
_output_shapes
:2
y
 bilm/CNN_1/Conv2D/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_0*
dtype0*&
_output_shapes
: 
╗
bilm/CNN_1/Conv2DConv2D"bilm/embedding_lookup_1/Identity_1 bilm/CNN_1/Conv2D/ReadVariableOp*&
_output_shapes
:2 *
T0*
strides
*
paddingVALID
j
bilm/CNN_1/add/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_0*
dtype0*
_output_shapes
: 
z
bilm/CNN_1/addAddV2bilm/CNN_1/Conv2Dbilm/CNN_1/add/ReadVariableOp*
T0*&
_output_shapes
:2 
Љ
bilm/CNN_1/MaxPoolMaxPoolbilm/CNN_1/add*
strides
*
paddingVALID*
ksize
2*&
_output_shapes
: 
\
bilm/CNN_1/ReluRelubilm/CNN_1/MaxPool*
T0*&
_output_shapes
: 
r
bilm/CNN_1/SqueezeSqueezebilm/CNN_1/Relu*
T0*
squeeze_dims
*"
_output_shapes
: 
{
"bilm/CNN_1/Conv2D_1/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_1*
dtype0*&
_output_shapes
: 
┐
bilm/CNN_1/Conv2D_1Conv2D"bilm/embedding_lookup_1/Identity_1"bilm/CNN_1/Conv2D_1/ReadVariableOp*&
_output_shapes
:1 *
strides
*
T0*
paddingVALID
l
bilm/CNN_1/add_1/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_1*
dtype0*
_output_shapes
: 
ђ
bilm/CNN_1/add_1AddV2bilm/CNN_1/Conv2D_1bilm/CNN_1/add_1/ReadVariableOp*
T0*&
_output_shapes
:1 
Ћ
bilm/CNN_1/MaxPool_1MaxPoolbilm/CNN_1/add_1*
ksize
1*&
_output_shapes
: *
strides
*
paddingVALID
`
bilm/CNN_1/Relu_1Relubilm/CNN_1/MaxPool_1*
T0*&
_output_shapes
: 
v
bilm/CNN_1/Squeeze_1Squeezebilm/CNN_1/Relu_1*
T0*
squeeze_dims
*"
_output_shapes
: 
{
"bilm/CNN_1/Conv2D_2/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_2*
dtype0*&
_output_shapes
:@
┐
bilm/CNN_1/Conv2D_2Conv2D"bilm/embedding_lookup_1/Identity_1"bilm/CNN_1/Conv2D_2/ReadVariableOp*
T0*
strides
*
paddingVALID*&
_output_shapes
:0@
l
bilm/CNN_1/add_2/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_2*
dtype0*
_output_shapes
:@
ђ
bilm/CNN_1/add_2AddV2bilm/CNN_1/Conv2D_2bilm/CNN_1/add_2/ReadVariableOp*
T0*&
_output_shapes
:0@
Ћ
bilm/CNN_1/MaxPool_2MaxPoolbilm/CNN_1/add_2*&
_output_shapes
:@*
strides
*
paddingVALID*
ksize
0
`
bilm/CNN_1/Relu_2Relubilm/CNN_1/MaxPool_2*&
_output_shapes
:@*
T0
v
bilm/CNN_1/Squeeze_2Squeezebilm/CNN_1/Relu_2*
T0*
squeeze_dims
*"
_output_shapes
:@
|
"bilm/CNN_1/Conv2D_3/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_3*
dtype0*'
_output_shapes
:ђ
└
bilm/CNN_1/Conv2D_3Conv2D"bilm/embedding_lookup_1/Identity_1"bilm/CNN_1/Conv2D_3/ReadVariableOp*
strides
*
T0*
paddingVALID*'
_output_shapes
:/ђ
m
bilm/CNN_1/add_3/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_3*
dtype0*
_output_shapes	
:ђ
Ђ
bilm/CNN_1/add_3AddV2bilm/CNN_1/Conv2D_3bilm/CNN_1/add_3/ReadVariableOp*
T0*'
_output_shapes
:/ђ
ќ
bilm/CNN_1/MaxPool_3MaxPoolbilm/CNN_1/add_3*
ksize
/*'
_output_shapes
:ђ*
strides
*
paddingVALID
a
bilm/CNN_1/Relu_3Relubilm/CNN_1/MaxPool_3*'
_output_shapes
:ђ*
T0
w
bilm/CNN_1/Squeeze_3Squeezebilm/CNN_1/Relu_3*
T0*
squeeze_dims
*#
_output_shapes
:ђ
|
"bilm/CNN_1/Conv2D_4/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_4*
dtype0*'
_output_shapes
:ђ
└
bilm/CNN_1/Conv2D_4Conv2D"bilm/embedding_lookup_1/Identity_1"bilm/CNN_1/Conv2D_4/ReadVariableOp*'
_output_shapes
:.ђ*
strides
*
T0*
paddingVALID
m
bilm/CNN_1/add_4/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_4*
dtype0*
_output_shapes	
:ђ
Ђ
bilm/CNN_1/add_4AddV2bilm/CNN_1/Conv2D_4bilm/CNN_1/add_4/ReadVariableOp*'
_output_shapes
:.ђ*
T0
ќ
bilm/CNN_1/MaxPool_4MaxPoolbilm/CNN_1/add_4*
strides
*
paddingVALID*
ksize
.*'
_output_shapes
:ђ
a
bilm/CNN_1/Relu_4Relubilm/CNN_1/MaxPool_4*'
_output_shapes
:ђ*
T0
w
bilm/CNN_1/Squeeze_4Squeezebilm/CNN_1/Relu_4*
T0*
squeeze_dims
*#
_output_shapes
:ђ
|
"bilm/CNN_1/Conv2D_5/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_5*
dtype0*'
_output_shapes
:ђ
└
bilm/CNN_1/Conv2D_5Conv2D"bilm/embedding_lookup_1/Identity_1"bilm/CNN_1/Conv2D_5/ReadVariableOp*
T0*
strides
*
paddingVALID*'
_output_shapes
:-ђ
m
bilm/CNN_1/add_5/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_5*
dtype0*
_output_shapes	
:ђ
Ђ
bilm/CNN_1/add_5AddV2bilm/CNN_1/Conv2D_5bilm/CNN_1/add_5/ReadVariableOp*'
_output_shapes
:-ђ*
T0
ќ
bilm/CNN_1/MaxPool_5MaxPoolbilm/CNN_1/add_5*
ksize
-*'
_output_shapes
:ђ*
strides
*
paddingVALID
a
bilm/CNN_1/Relu_5Relubilm/CNN_1/MaxPool_5*
T0*'
_output_shapes
:ђ
w
bilm/CNN_1/Squeeze_5Squeezebilm/CNN_1/Relu_5*#
_output_shapes
:ђ*
T0*
squeeze_dims

|
"bilm/CNN_1/Conv2D_6/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_6*
dtype0*'
_output_shapes
:ђ
└
bilm/CNN_1/Conv2D_6Conv2D"bilm/embedding_lookup_1/Identity_1"bilm/CNN_1/Conv2D_6/ReadVariableOp*
T0*
strides
*
paddingVALID*'
_output_shapes
:,ђ
m
bilm/CNN_1/add_6/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_6*
dtype0*
_output_shapes	
:ђ
Ђ
bilm/CNN_1/add_6AddV2bilm/CNN_1/Conv2D_6bilm/CNN_1/add_6/ReadVariableOp*
T0*'
_output_shapes
:,ђ
ќ
bilm/CNN_1/MaxPool_6MaxPoolbilm/CNN_1/add_6*
ksize
,*'
_output_shapes
:ђ*
strides
*
paddingVALID
a
bilm/CNN_1/Relu_6Relubilm/CNN_1/MaxPool_6*
T0*'
_output_shapes
:ђ
w
bilm/CNN_1/Squeeze_6Squeezebilm/CNN_1/Relu_6*
squeeze_dims
*#
_output_shapes
:ђ*
T0
T
bilm/concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Ч
bilm/concat_1ConcatV2bilm/CNN_1/Squeezebilm/CNN_1/Squeeze_1bilm/CNN_1/Squeeze_2bilm/CNN_1/Squeeze_3bilm/CNN_1/Squeeze_4bilm/CNN_1/Squeeze_5bilm/CNN_1/Squeeze_6bilm/concat_1/axis*
N*
T0*#
_output_shapes
:ђ
a
bilm/Shape_1Const*!
valueB"         *
dtype0*
_output_shapes
:
e
bilm/Reshape_2/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
h
bilm/Reshape_2Reshapebilm/concat_1bilm/Reshape_2/shape*
_output_shapes
:	ђ*
T0
v
bilm/MatMul_5/ReadVariableOpReadVariableOpbilm/CNN_high_0/W_carry*
dtype0* 
_output_shapes
:
ђђ
o
bilm/MatMul_5MatMulbilm/Reshape_2bilm/MatMul_5/ReadVariableOp*
T0*
_output_shapes
:	ђ
n
bilm/add_7/ReadVariableOpReadVariableOpbilm/CNN_high_0/b_carry*
dtype0*
_output_shapes	
:ђ
g

bilm/add_7AddV2bilm/MatMul_5bilm/add_7/ReadVariableOp*
_output_shapes
:	ђ*
T0
O
bilm/Sigmoid_2Sigmoid
bilm/add_7*
_output_shapes
:	ђ*
T0
z
bilm/MatMul_6/ReadVariableOpReadVariableOpbilm/CNN_high_0/W_transform*
dtype0* 
_output_shapes
:
ђђ
o
bilm/MatMul_6MatMulbilm/Reshape_2bilm/MatMul_6/ReadVariableOp*
_output_shapes
:	ђ*
T0
r
bilm/add_8/ReadVariableOpReadVariableOpbilm/CNN_high_0/b_transform*
dtype0*
_output_shapes	
:ђ
g

bilm/add_8AddV2bilm/MatMul_6bilm/add_8/ReadVariableOp*
_output_shapes
:	ђ*
T0
I
bilm/Relu_2Relu
bilm/add_8*
T0*
_output_shapes
:	ђ
X

bilm/mul_5Mulbilm/Sigmoid_2bilm/Relu_2*
_output_shapes
:	ђ*
T0
Q
bilm/sub_2/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Y

bilm/sub_2Subbilm/sub_2/xbilm/Sigmoid_2*
_output_shapes
:	ђ*
T0
W

bilm/mul_6Mul
bilm/sub_2bilm/Reshape_2*
T0*
_output_shapes
:	ђ
U

bilm/add_9AddV2
bilm/mul_5
bilm/mul_6*
_output_shapes
:	ђ*
T0
v
bilm/MatMul_7/ReadVariableOpReadVariableOpbilm/CNN_high_1/W_carry*
dtype0* 
_output_shapes
:
ђђ
k
bilm/MatMul_7MatMul
bilm/add_9bilm/MatMul_7/ReadVariableOp*
_output_shapes
:	ђ*
T0
o
bilm/add_10/ReadVariableOpReadVariableOpbilm/CNN_high_1/b_carry*
dtype0*
_output_shapes	
:ђ
i
bilm/add_10AddV2bilm/MatMul_7bilm/add_10/ReadVariableOp*
T0*
_output_shapes
:	ђ
P
bilm/Sigmoid_3Sigmoidbilm/add_10*
_output_shapes
:	ђ*
T0
z
bilm/MatMul_8/ReadVariableOpReadVariableOpbilm/CNN_high_1/W_transform*
dtype0* 
_output_shapes
:
ђђ
k
bilm/MatMul_8MatMul
bilm/add_9bilm/MatMul_8/ReadVariableOp*
_output_shapes
:	ђ*
T0
s
bilm/add_11/ReadVariableOpReadVariableOpbilm/CNN_high_1/b_transform*
dtype0*
_output_shapes	
:ђ
i
bilm/add_11AddV2bilm/MatMul_8bilm/add_11/ReadVariableOp*
T0*
_output_shapes
:	ђ
J
bilm/Relu_3Relubilm/add_11*
_output_shapes
:	ђ*
T0
X

bilm/mul_7Mulbilm/Sigmoid_3bilm/Relu_3*
_output_shapes
:	ђ*
T0
Q
bilm/sub_3/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Y

bilm/sub_3Subbilm/sub_3/xbilm/Sigmoid_3*
T0*
_output_shapes
:	ђ
S

bilm/mul_8Mul
bilm/sub_3
bilm/add_9*
_output_shapes
:	ђ*
T0
V
bilm/add_12AddV2
bilm/mul_7
bilm/mul_8*
T0*
_output_shapes
:	ђ
s
bilm/MatMul_9/ReadVariableOpReadVariableOpbilm/CNN_proj/W_proj*
dtype0* 
_output_shapes
:
ђђ
l
bilm/MatMul_9MatMulbilm/add_12bilm/MatMul_9/ReadVariableOp*
T0*
_output_shapes
:	ђ
l
bilm/add_13/ReadVariableOpReadVariableOpbilm/CNN_proj/b_proj*
dtype0*
_output_shapes	
:ђ
i
bilm/add_13AddV2bilm/MatMul_9bilm/add_13/ReadVariableOp*
_output_shapes
:	ђ*
T0
i
bilm/Reshape_3/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
j
bilm/Reshape_3Reshapebilm/add_13bilm/Reshape_3/shape*#
_output_shapes
:ђ*
T0
Х
bilm/embedding_lookup_2ResourceGatherbilm/char_embedbilm/ExpandDims_3*"
_class
loc:@bilm/char_embed*
dtype0*
Tindices0*&
_output_shapes
:2
џ
 bilm/embedding_lookup_2/IdentityIdentitybilm/embedding_lookup_2*
T0*"
_class
loc:@bilm/char_embed*&
_output_shapes
:2
Ђ
"bilm/embedding_lookup_2/Identity_1Identity bilm/embedding_lookup_2/Identity*
T0*&
_output_shapes
:2
y
 bilm/CNN_2/Conv2D/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_0*
dtype0*&
_output_shapes
: 
╗
bilm/CNN_2/Conv2DConv2D"bilm/embedding_lookup_2/Identity_1 bilm/CNN_2/Conv2D/ReadVariableOp*&
_output_shapes
:2 *
T0*
strides
*
paddingVALID
j
bilm/CNN_2/add/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_0*
dtype0*
_output_shapes
: 
z
bilm/CNN_2/addAddV2bilm/CNN_2/Conv2Dbilm/CNN_2/add/ReadVariableOp*
T0*&
_output_shapes
:2 
Љ
bilm/CNN_2/MaxPoolMaxPoolbilm/CNN_2/add*
strides
*
paddingVALID*
ksize
2*&
_output_shapes
: 
\
bilm/CNN_2/ReluRelubilm/CNN_2/MaxPool*
T0*&
_output_shapes
: 
r
bilm/CNN_2/SqueezeSqueezebilm/CNN_2/Relu*
squeeze_dims
*"
_output_shapes
: *
T0
{
"bilm/CNN_2/Conv2D_1/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_1*
dtype0*&
_output_shapes
: 
┐
bilm/CNN_2/Conv2D_1Conv2D"bilm/embedding_lookup_2/Identity_1"bilm/CNN_2/Conv2D_1/ReadVariableOp*&
_output_shapes
:1 *
T0*
strides
*
paddingVALID
l
bilm/CNN_2/add_1/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_1*
dtype0*
_output_shapes
: 
ђ
bilm/CNN_2/add_1AddV2bilm/CNN_2/Conv2D_1bilm/CNN_2/add_1/ReadVariableOp*
T0*&
_output_shapes
:1 
Ћ
bilm/CNN_2/MaxPool_1MaxPoolbilm/CNN_2/add_1*
ksize
1*&
_output_shapes
: *
strides
*
paddingVALID
`
bilm/CNN_2/Relu_1Relubilm/CNN_2/MaxPool_1*&
_output_shapes
: *
T0
v
bilm/CNN_2/Squeeze_1Squeezebilm/CNN_2/Relu_1*
T0*
squeeze_dims
*"
_output_shapes
: 
{
"bilm/CNN_2/Conv2D_2/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_2*
dtype0*&
_output_shapes
:@
┐
bilm/CNN_2/Conv2D_2Conv2D"bilm/embedding_lookup_2/Identity_1"bilm/CNN_2/Conv2D_2/ReadVariableOp*&
_output_shapes
:0@*
T0*
strides
*
paddingVALID
l
bilm/CNN_2/add_2/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_2*
dtype0*
_output_shapes
:@
ђ
bilm/CNN_2/add_2AddV2bilm/CNN_2/Conv2D_2bilm/CNN_2/add_2/ReadVariableOp*
T0*&
_output_shapes
:0@
Ћ
bilm/CNN_2/MaxPool_2MaxPoolbilm/CNN_2/add_2*&
_output_shapes
:@*
strides
*
paddingVALID*
ksize
0
`
bilm/CNN_2/Relu_2Relubilm/CNN_2/MaxPool_2*
T0*&
_output_shapes
:@
v
bilm/CNN_2/Squeeze_2Squeezebilm/CNN_2/Relu_2*
T0*
squeeze_dims
*"
_output_shapes
:@
|
"bilm/CNN_2/Conv2D_3/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_3*
dtype0*'
_output_shapes
:ђ
└
bilm/CNN_2/Conv2D_3Conv2D"bilm/embedding_lookup_2/Identity_1"bilm/CNN_2/Conv2D_3/ReadVariableOp*
strides
*
T0*
paddingVALID*'
_output_shapes
:/ђ
m
bilm/CNN_2/add_3/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_3*
dtype0*
_output_shapes	
:ђ
Ђ
bilm/CNN_2/add_3AddV2bilm/CNN_2/Conv2D_3bilm/CNN_2/add_3/ReadVariableOp*'
_output_shapes
:/ђ*
T0
ќ
bilm/CNN_2/MaxPool_3MaxPoolbilm/CNN_2/add_3*
ksize
/*'
_output_shapes
:ђ*
strides
*
paddingVALID
a
bilm/CNN_2/Relu_3Relubilm/CNN_2/MaxPool_3*
T0*'
_output_shapes
:ђ
w
bilm/CNN_2/Squeeze_3Squeezebilm/CNN_2/Relu_3*#
_output_shapes
:ђ*
T0*
squeeze_dims

|
"bilm/CNN_2/Conv2D_4/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_4*
dtype0*'
_output_shapes
:ђ
└
bilm/CNN_2/Conv2D_4Conv2D"bilm/embedding_lookup_2/Identity_1"bilm/CNN_2/Conv2D_4/ReadVariableOp*'
_output_shapes
:.ђ*
strides
*
T0*
paddingVALID
m
bilm/CNN_2/add_4/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_4*
dtype0*
_output_shapes	
:ђ
Ђ
bilm/CNN_2/add_4AddV2bilm/CNN_2/Conv2D_4bilm/CNN_2/add_4/ReadVariableOp*
T0*'
_output_shapes
:.ђ
ќ
bilm/CNN_2/MaxPool_4MaxPoolbilm/CNN_2/add_4*
strides
*
paddingVALID*
ksize
.*'
_output_shapes
:ђ
a
bilm/CNN_2/Relu_4Relubilm/CNN_2/MaxPool_4*'
_output_shapes
:ђ*
T0
w
bilm/CNN_2/Squeeze_4Squeezebilm/CNN_2/Relu_4*
T0*
squeeze_dims
*#
_output_shapes
:ђ
|
"bilm/CNN_2/Conv2D_5/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_5*
dtype0*'
_output_shapes
:ђ
└
bilm/CNN_2/Conv2D_5Conv2D"bilm/embedding_lookup_2/Identity_1"bilm/CNN_2/Conv2D_5/ReadVariableOp*
strides
*
T0*
paddingVALID*'
_output_shapes
:-ђ
m
bilm/CNN_2/add_5/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_5*
dtype0*
_output_shapes	
:ђ
Ђ
bilm/CNN_2/add_5AddV2bilm/CNN_2/Conv2D_5bilm/CNN_2/add_5/ReadVariableOp*
T0*'
_output_shapes
:-ђ
ќ
bilm/CNN_2/MaxPool_5MaxPoolbilm/CNN_2/add_5*
strides
*
paddingVALID*
ksize
-*'
_output_shapes
:ђ
a
bilm/CNN_2/Relu_5Relubilm/CNN_2/MaxPool_5*'
_output_shapes
:ђ*
T0
w
bilm/CNN_2/Squeeze_5Squeezebilm/CNN_2/Relu_5*
T0*
squeeze_dims
*#
_output_shapes
:ђ
|
"bilm/CNN_2/Conv2D_6/ReadVariableOpReadVariableOpbilm/CNN/W_cnn_6*
dtype0*'
_output_shapes
:ђ
└
bilm/CNN_2/Conv2D_6Conv2D"bilm/embedding_lookup_2/Identity_1"bilm/CNN_2/Conv2D_6/ReadVariableOp*'
_output_shapes
:,ђ*
T0*
strides
*
paddingVALID
m
bilm/CNN_2/add_6/ReadVariableOpReadVariableOpbilm/CNN/b_cnn_6*
dtype0*
_output_shapes	
:ђ
Ђ
bilm/CNN_2/add_6AddV2bilm/CNN_2/Conv2D_6bilm/CNN_2/add_6/ReadVariableOp*
T0*'
_output_shapes
:,ђ
ќ
bilm/CNN_2/MaxPool_6MaxPoolbilm/CNN_2/add_6*
strides
*
paddingVALID*
ksize
,*'
_output_shapes
:ђ
a
bilm/CNN_2/Relu_6Relubilm/CNN_2/MaxPool_6*'
_output_shapes
:ђ*
T0
w
bilm/CNN_2/Squeeze_6Squeezebilm/CNN_2/Relu_6*
T0*
squeeze_dims
*#
_output_shapes
:ђ
T
bilm/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B :
Ч
bilm/concat_2ConcatV2bilm/CNN_2/Squeezebilm/CNN_2/Squeeze_1bilm/CNN_2/Squeeze_2bilm/CNN_2/Squeeze_3bilm/CNN_2/Squeeze_4bilm/CNN_2/Squeeze_5bilm/CNN_2/Squeeze_6bilm/concat_2/axis*#
_output_shapes
:ђ*
N*
T0
a
bilm/Shape_2Const*!
valueB"         *
dtype0*
_output_shapes
:
e
bilm/Reshape_4/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
h
bilm/Reshape_4Reshapebilm/concat_2bilm/Reshape_4/shape*
T0*
_output_shapes
:	ђ
w
bilm/MatMul_10/ReadVariableOpReadVariableOpbilm/CNN_high_0/W_carry*
dtype0* 
_output_shapes
:
ђђ
q
bilm/MatMul_10MatMulbilm/Reshape_4bilm/MatMul_10/ReadVariableOp*
_output_shapes
:	ђ*
T0
o
bilm/add_14/ReadVariableOpReadVariableOpbilm/CNN_high_0/b_carry*
dtype0*
_output_shapes	
:ђ
j
bilm/add_14AddV2bilm/MatMul_10bilm/add_14/ReadVariableOp*
T0*
_output_shapes
:	ђ
P
bilm/Sigmoid_4Sigmoidbilm/add_14*
T0*
_output_shapes
:	ђ
{
bilm/MatMul_11/ReadVariableOpReadVariableOpbilm/CNN_high_0/W_transform*
dtype0* 
_output_shapes
:
ђђ
q
bilm/MatMul_11MatMulbilm/Reshape_4bilm/MatMul_11/ReadVariableOp*
T0*
_output_shapes
:	ђ
s
bilm/add_15/ReadVariableOpReadVariableOpbilm/CNN_high_0/b_transform*
dtype0*
_output_shapes	
:ђ
j
bilm/add_15AddV2bilm/MatMul_11bilm/add_15/ReadVariableOp*
T0*
_output_shapes
:	ђ
J
bilm/Relu_4Relubilm/add_15*
T0*
_output_shapes
:	ђ
X

bilm/mul_9Mulbilm/Sigmoid_4bilm/Relu_4*
T0*
_output_shapes
:	ђ
Q
bilm/sub_4/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
Y

bilm/sub_4Subbilm/sub_4/xbilm/Sigmoid_4*
_output_shapes
:	ђ*
T0
X
bilm/mul_10Mul
bilm/sub_4bilm/Reshape_4*
T0*
_output_shapes
:	ђ
W
bilm/add_16AddV2
bilm/mul_9bilm/mul_10*
T0*
_output_shapes
:	ђ
w
bilm/MatMul_12/ReadVariableOpReadVariableOpbilm/CNN_high_1/W_carry*
dtype0* 
_output_shapes
:
ђђ
n
bilm/MatMul_12MatMulbilm/add_16bilm/MatMul_12/ReadVariableOp*
T0*
_output_shapes
:	ђ
o
bilm/add_17/ReadVariableOpReadVariableOpbilm/CNN_high_1/b_carry*
dtype0*
_output_shapes	
:ђ
j
bilm/add_17AddV2bilm/MatMul_12bilm/add_17/ReadVariableOp*
T0*
_output_shapes
:	ђ
P
bilm/Sigmoid_5Sigmoidbilm/add_17*
T0*
_output_shapes
:	ђ
{
bilm/MatMul_13/ReadVariableOpReadVariableOpbilm/CNN_high_1/W_transform*
dtype0* 
_output_shapes
:
ђђ
n
bilm/MatMul_13MatMulbilm/add_16bilm/MatMul_13/ReadVariableOp*
T0*
_output_shapes
:	ђ
s
bilm/add_18/ReadVariableOpReadVariableOpbilm/CNN_high_1/b_transform*
dtype0*
_output_shapes	
:ђ
j
bilm/add_18AddV2bilm/MatMul_13bilm/add_18/ReadVariableOp*
T0*
_output_shapes
:	ђ
J
bilm/Relu_5Relubilm/add_18*
T0*
_output_shapes
:	ђ
Y
bilm/mul_11Mulbilm/Sigmoid_5bilm/Relu_5*
T0*
_output_shapes
:	ђ
Q
bilm/sub_5/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
Y

bilm/sub_5Subbilm/sub_5/xbilm/Sigmoid_5*
_output_shapes
:	ђ*
T0
U
bilm/mul_12Mul
bilm/sub_5bilm/add_16*
T0*
_output_shapes
:	ђ
X
bilm/add_19AddV2bilm/mul_11bilm/mul_12*
T0*
_output_shapes
:	ђ
t
bilm/MatMul_14/ReadVariableOpReadVariableOpbilm/CNN_proj/W_proj*
dtype0* 
_output_shapes
:
ђђ
n
bilm/MatMul_14MatMulbilm/add_19bilm/MatMul_14/ReadVariableOp*
T0*
_output_shapes
:	ђ
l
bilm/add_20/ReadVariableOpReadVariableOpbilm/CNN_proj/b_proj*
dtype0*
_output_shapes	
:ђ
j
bilm/add_20AddV2bilm/MatMul_14bilm/add_20/ReadVariableOp*
_output_shapes
:	ђ*
T0
i
bilm/Reshape_5/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
j
bilm/Reshape_5Reshapebilm/add_20bilm/Reshape_5/shape*
T0*#
_output_shapes
:ђ
J
bilm/Shape_3Shapebilm/Reshape_1*
_output_shapes
:*
T0
d
bilm/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: 
f
bilm/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
f
bilm/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
л
bilm/strided_slice_2StridedSlicebilm/Shape_3bilm/strided_slice_2/stackbilm/strided_slice_2/stack_1bilm/strided_slice_2/stack_2*
T0*
_output_shapes
: *
Index0*
shrink_axis_mask
W
bilm/Tile/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
W
bilm/Tile/multiples/2Const*
value	B :*
dtype0*
_output_shapes
: 
Ї
bilm/Tile/multiplesPackbilm/strided_slice_2bilm/Tile/multiples/1bilm/Tile/multiples/2*
N*
T0*
_output_shapes
:
m
	bilm/TileTilebilm/Reshape_3bilm/Tile/multiples*,
_output_shapes
:         ђ*
T0
Y
bilm/Tile_1/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
Y
bilm/Tile_1/multiples/2Const*
value	B :*
dtype0*
_output_shapes
: 
Њ
bilm/Tile_1/multiplesPackbilm/strided_slice_2bilm/Tile_1/multiples/1bilm/Tile_1/multiples/2*
_output_shapes
:*
N*
T0
q
bilm/Tile_1Tilebilm/Reshape_5bilm/Tile_1/multiples*,
_output_shapes
:         ђ*
T0
T
bilm/concat_3/axisConst*
dtype0*
_output_shapes
: *
value	B :
Љ
bilm/concat_3ConcatV2	bilm/Tilebilm/Reshape_1bilm/concat_3/axis*5
_output_shapes#
!:                  ђ*
N*
T0
i
'bilm/RNN_0/RNN/MultiRNNCell/Cell0/add/yConst*
dtype0*
_output_shapes
: *
value	B :
і
%bilm/RNN_0/RNN/MultiRNNCell/Cell0/addAddV2Sum'bilm/RNN_0/RNN/MultiRNNCell/Cell0/add/y*#
_output_shapes
:         *
T0
l
*bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
s
1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
s
1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Р
+bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/rangeRange1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/range/start*bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Rank1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/range/delta*
_output_shapes
:
є
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
s
1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
§
,bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/concatConcatV25bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/concat/values_0+bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/range1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/concat/axis*
N*
T0*
_output_shapes
:
╣
/bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/transpose	Transposebilm/concat_3,bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/concat*5
_output_shapes#
!:                  ђ*
T0
ќ
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/sequence_lengthIdentity%bilm/RNN_0/RNN/MultiRNNCell/Cell0/add*#
_output_shapes
:         *
T0
і
+bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/ShapeShape/bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/transpose*
T0*
_output_shapes
:
Ѓ
9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
Ё
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ё
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
в
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_sliceStridedSlice+bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Shape9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice/stack;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice/stack_1;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice/stack_2*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0
ѕ
Fbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
Ы
Bbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims
ExpandDims3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_sliceFbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims/dim*
T0*
_output_shapes
:
ѕ
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ConstConst*
valueB:ђ *
dtype0*
_output_shapes
:
Ё
Cbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
└
>bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/concatConcatV2Bbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ConstCbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/concat/axis*
N*
T0*
_output_shapes
:
ѕ
Cbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/zerosFill>bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/concatCbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/zeros/Const*(
_output_shapes
:         ђ *
T0
і
Hbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 
Ш
Dbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_1
ExpandDims3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_sliceHbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_1/dim*
_output_shapes
:*
T0
і
?bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/Const_1Const*
valueB:ђ *
dtype0*
_output_shapes
:
і
Hbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ш
Dbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_2
ExpandDims3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_sliceHbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_2/dim*
_output_shapes
:*
T0
і
?bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/Const_2Const*
dtype0*
_output_shapes
:*
valueB:ђ
Є
Ebilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
╚
@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/concat_1ConcatV2Dbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_2?bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/Const_2Ebilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/concat_1/axis*
N*
T0*
_output_shapes
:
і
Ebilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ѓ
?bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/zeros_1Fill@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/concat_1Ebilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/zeros_1/Const*(
_output_shapes
:         ђ*
T0
і
Hbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ш
Dbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_3
ExpandDims3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_sliceHbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_3/dim*
T0*
_output_shapes
:
і
?bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/Const_3Const*
valueB:ђ*
dtype0*
_output_shapes
:
њ
-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Shape_1Shape5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/sequence_length*
T0*
_output_shapes
:
ќ
+bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/stackPack3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice*
N*
T0*
_output_shapes
:
х
+bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/EqualEqual-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Shape_1+bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/stack*
T0*
_output_shapes
:
u
+bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
б
)bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/AllAll+bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Equal+bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Const*
_output_shapes
: 
╚
2bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Assert/ConstConst*
dtype0*
_output_shapes
: *f
value]B[ BUExpected shape for Tensor bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/sequence_length:0 is 
Ё
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
л
:bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Assert/Assert/data_0Const*f
value]B[ BUExpected shape for Tensor bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
І
:bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
╦
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Assert/AssertAssert)bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/All:bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Assert/Assert/data_0+bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/stack:bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Assert/Assert/data_2-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Shape_1*
T
2
п
1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/CheckSeqLenIdentity5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/sequence_length4^bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Assert/Assert*
T0*#
_output_shapes
:         
ї
-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Shape_2Shape/bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/transpose*
T0*
_output_shapes
:
Ё
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
Є
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Є
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ш
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1StridedSlice-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Shape_2;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1/stack=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1/stack_1=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1/stack_2*
T0*
_output_shapes
: *
Index0*
shrink_axis_mask
ї
-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Shape_3Shape/bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/transpose*
T0*
_output_shapes
:
Ё
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
Є
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Є
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ш
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_2StridedSlice-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Shape_3;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_2/stack=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_2/stack_1=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_2/stack_2*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0
v
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
л
0bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/ExpandDims
ExpandDims5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_24bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/ExpandDims/dim*
T0*
_output_shapes
:
x
-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Const_1Const*
valueB:ђ*
dtype0*
_output_shapes
:
u
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
■
.bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/concat_1ConcatV20bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/ExpandDims-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Const_13bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/concat_1/axis*
_output_shapes
:*
N*
T0
v
1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╔
+bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/zerosFill.bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/concat_11bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/zeros/Const*(
_output_shapes
:         ђ*
T0
w
-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
│
)bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/MinMin1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/CheckSeqLen-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Const_2*
_output_shapes
: *
T0
w
-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
│
)bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/MaxMax1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/CheckSeqLen-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Const_3*
T0*
_output_shapes
: 
l
*bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
Й
1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayTensorArrayV35bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1*%
element_shape:         ђ*
identical_element_shapes(*
dtype0*
_output_shapes

:: *Q
tensor_array_name<:bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/dynamic_rnn/output_0
┐
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArray_1TensorArrayV35bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1*P
tensor_array_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/dynamic_rnn/input_0*%
element_shape:         ђ*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
Ю
>bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/ShapeShape/bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/transpose*
T0*
_output_shapes
:
ќ
Lbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
ў
Nbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
ў
Nbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╩
Fbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/strided_sliceStridedSlice>bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/ShapeLbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/strided_slice/stackNbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/strided_slice/stack_1Nbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/strided_slice/stack_2*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: 
є
Dbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
є
Dbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
└
>bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/rangeRangeDbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/range/startFbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/strided_sliceDbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:         
║
`bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV33bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArray_1>bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/range/bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/transpose5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArray_1:1*B
_class8
64loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/transpose*
_output_shapes
: *
T0
q
/bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
х
-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/MaximumMaximum/bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Maximum/x)bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Max*
_output_shapes
: *
T0
┐
-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/MinimumMinimum5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Maximum*
T0*
_output_shapes
: 

=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
ђ
1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/EnterEnter=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/iteration_counter*
T0*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
_output_shapes
: *
parallel_iterations 
№
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter_1Enter*bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/time*
parallel_iterations *
T0*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
_output_shapes
: 
Э
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter_2Enter3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArray:1*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
_output_shapes
: *
parallel_iterations *
T0
ћ
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter_3Enter=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/zeros*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/while_context*(
_output_shapes
:         ђ *
parallel_iterations *
T0
ќ
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter_4Enter?bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/zeros_1*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/while_context*(
_output_shapes
:         ђ*
parallel_iterations *
T0
н
1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/MergeMerge1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration*
N*
T0*
_output_shapes
: : 
┌
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_1Merge3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter_1;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_1*
N*
T0*
_output_shapes
: : 
┌
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_2Merge3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter_2;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_2*
N*
T0*
_output_shapes
: : 
В
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_3Merge3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter_3;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_3*
N*
T0**
_output_shapes
:         ђ : 
В
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_4Merge3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter_4;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_4*
N*
T0**
_output_shapes
:         ђ: 
─
0bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/LessLess1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Less/Enter*
_output_shapes
: *
T0
љ
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Less/EnterEnter5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1*
T0*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*
_output_shapes
: *
parallel_iterations 
╩
2bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Less_1Less3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_18bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Less_1/Enter*
_output_shapes
: *
T0
і
8bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Less_1/EnterEnter-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Minimum*
T0*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*
_output_shapes
: *
parallel_iterations 
┬
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/LogicalAnd
LogicalAnd0bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Less2bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Less_1*
_output_shapes
: 
љ
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/LoopCondLoopCond6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/LogicalAnd*
_output_shapes
: 
ј
2bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/SwitchSwitch1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/LoopCond*
T0*D
_class:
86loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge*
_output_shapes
: : 
ћ
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_1Switch3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_14bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/LoopCond*
T0*F
_class<
:8loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_1*
_output_shapes
: : 
ћ
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_2Switch3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_24bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/LoopCond*
T0*F
_class<
:8loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_2*
_output_shapes
: : 
И
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_3Switch3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_34bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/LoopCond*
T0*F
_class<
:8loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_3*<
_output_shapes*
(:         ђ :         ђ 
И
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_4Switch3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_44bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/LoopCond*
T0*F
_class<
:8loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_4*<
_output_shapes*
(:         ђ:         ђ
Ќ
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/IdentityIdentity4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch:1*
T0*
_output_shapes
: 
Џ
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_1Identity6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_1:1*
_output_shapes
: *
T0
Џ
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_2Identity6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
Г
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_3Identity6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_3:1*(
_output_shapes
:         ђ *
T0
Г
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_4Identity6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_4:1*
T0*(
_output_shapes
:         ђ
ф
1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/add/yConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
┬
/bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/addAddV24bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/add/y*
_output_shapes
: *
T0
═
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3TensorArrayReadV3Cbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3/Enter6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_1Ebilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:         ђ
Ъ
Cbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3/EnterEnter3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArray_1*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*
_output_shapes
:*
parallel_iterations *
T0
╩
Ebilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3/Enter_1Enter`bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*
_output_shapes
: *
parallel_iterations *
T0
Ь
8bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqualGreaterEqual6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_1>bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqual/Enter*
T0*#
_output_shapes
:         
А
>bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqual/EnterEnter1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/CheckSeqLen*
parallel_iterations *
T0*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*#
_output_shapes
:         
з
Wbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"    @  *I
_class?
=;loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel
т
Ubilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *0ў╝*I
_class?
=;loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*
dtype0*
_output_shapes
: 
т
Ubilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *0ў<*I
_class?
=;loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*
dtype0*
_output_shapes
: 
═
_bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformWbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/shape*I
_class?
=;loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*
dtype0*!
_output_shapes
:ђђђ*
T0
Ш
Ubilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/subSubUbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/maxUbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*I
_class?
=;loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel
І
Ubilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/mulMul_bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformUbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/sub*
T0*I
_class?
=;loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*!
_output_shapes
:ђђђ
§
Qbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniformAddUbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/mulUbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/min*I
_class?
=;loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*!
_output_shapes
:ђђђ*
T0
ќ
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernelVarHandleOp*
shape:ђђђ*I
_class?
=;loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*
dtype0*
_output_shapes
: *G
shared_name86bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel
й
Wbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*
_output_shapes
: 
ж
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/AssignAssignVariableOp6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernelQbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform*
dtype0
─
Jbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*
dtype0*!
_output_shapes
:ђђђ
╚
Dbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Read/IdentityIdentityJbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Read/ReadVariableOp*
T0*!
_output_shapes
:ђђђ
в
Vbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensorConst*
valueB:ђђ*G
_class=
;9loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias*
dtype0*
_output_shapes
:
┌
Lbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Initializer/zeros/ConstConst*
valueB
 *    *G
_class=
;9loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias*
dtype0*
_output_shapes
: 
С
Fbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Initializer/zerosFillVbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensorLbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Initializer/zeros/Const*
T0*G
_class=
;9loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias*
_output_shapes

:ђђ
І
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/biasVarHandleOp*
shape:ђђ*G
_class=
;9loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias*
dtype0*
_output_shapes
: *E
shared_name64bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias
╣
Ubilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias*
_output_shapes
: 
┌
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/AssignAssignVariableOp4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/biasFbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Initializer/zeros*
dtype0
╗
Hbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Read/ReadVariableOpReadVariableOp4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias*
dtype0*
_output_shapes

:ђђ
┐
Bbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Read/IdentityIdentityHbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Read/ReadVariableOp*
_output_shapes

:ђђ*
T0
Ѕ
bbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/shapeConst*
valueB"      *T
_classJ
HFloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel*
dtype0*
_output_shapes
:
ч
`bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/minConst*
valueB
 *:═й*T
_classJ
HFloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel*
dtype0*
_output_shapes
: 
ч
`bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/maxConst*
valueB
 *:═=*T
_classJ
HFloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel*
dtype0*
_output_shapes
: 
ь
jbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/RandomUniformRandomUniformbbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/shape*
T0*T
_classJ
HFloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel*
dtype0* 
_output_shapes
:
ђ ђ
б
`bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/subSub`bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/max`bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*T
_classJ
HFloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel
Х
`bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/mulMuljbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/RandomUniform`bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/sub*T
_classJ
HFloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel* 
_output_shapes
:
ђ ђ*
T0
е
\bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniformAdd`bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/mul`bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/min*
T0*T
_classJ
HFloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel* 
_output_shapes
:
ђ ђ
Х
Abilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernelVarHandleOp*
shape:
ђ ђ*T
_classJ
HFloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel*
dtype0*R
shared_nameCAbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel*
_output_shapes
: 
М
bbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpAbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel*
_output_shapes
: 
і
Hbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/AssignAssignVariableOpAbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel\bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform*
dtype0
┘
Ubilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Read/ReadVariableOpReadVariableOpAbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel*
dtype0* 
_output_shapes
:
ђ ђ
П
Obilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Read/IdentityIdentityUbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Read/ReadVariableOp*
T0* 
_output_shapes
:
ђ ђ
║
Abilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/concat/axisConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
Й
<bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/concatConcatV2=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV36bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_4Abilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/concat/axis*(
_output_shapes
:         ђ*
N*
T0
Ч
<bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMulMatMul<bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/concatBbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul/Enter*)
_output_shapes
:         ђђ*
T0
Х
Bbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul/EnterEnterDbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Read/Identity*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*!
_output_shapes
:ђђђ*
parallel_iterations *
T0
 
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/BiasAddBiasAdd<bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMulCbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/BiasAdd/Enter*
T0*)
_output_shapes
:         ђђ
░
Cbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/BiasAdd/EnterEnterBbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Read/Identity*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*
_output_shapes

:ђђ*
parallel_iterations *
T0
┤
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/ConstConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
Й
Ebilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split/split_dimConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╩
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/splitSplitEbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split/split_dim=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/BiasAdd*d
_output_shapesR
P:         ђ :         ђ :         ђ :         ђ *
T0*
	num_split
и
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/add/yConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
ы
9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/addAddV2=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split:2;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/add/y*
T0*(
_output_shapes
:         ђ 
Х
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/SigmoidSigmoid9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/add*(
_output_shapes
:         ђ *
T0
Ж
9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mulMul=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Sigmoid6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_3*(
_output_shapes
:         ђ *
T0
║
?bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Sigmoid_1Sigmoid;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split*(
_output_shapes
:         ђ *
T0
┤
:bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/TanhTanh=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split:1*(
_output_shapes
:         ђ *
T0
Ы
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mul_1Mul?bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Sigmoid_1:bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Tanh*(
_output_shapes
:         ђ *
T0
№
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/add_1AddV29bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mul;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mul_1*
T0*(
_output_shapes
:         ђ 
╔
Mbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/Minimum/yConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  @@
Ћ
Kbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/MinimumMinimum;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/add_1Mbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/Minimum/y*
T0*(
_output_shapes
:         ђ 
┴
Ebilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/yConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
valueB
 *  @└*
dtype0*
_output_shapes
: 
Ћ
Cbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_valueMaximumKbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/MinimumEbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/y*(
_output_shapes
:         ђ *
T0
╝
?bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Sigmoid_2Sigmoid=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split:3*(
_output_shapes
:         ђ *
T0
╝
<bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Tanh_1TanhCbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value*(
_output_shapes
:         ђ *
T0
З
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mul_2Mul?bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Sigmoid_2<bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Tanh_1*(
_output_shapes
:         ђ *
T0
■
>bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul_1MatMul;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mul_2Dbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul_1/Enter*(
_output_shapes
:         ђ*
T0
┬
Dbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul_1/EnterEnterObilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Read/Identity*
T0*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(* 
_output_shapes
:
ђ ђ*
parallel_iterations 
╦
Obilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/Minimum/yConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  @@
ю
Mbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/MinimumMinimum>bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul_1Obilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/Minimum/y*
T0*(
_output_shapes
:         ђ
├
Gbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/yConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
valueB
 *  @└*
dtype0*
_output_shapes
: 
Џ
Ebilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1MaximumMbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/MinimumGbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/y*(
_output_shapes
:         ђ*
T0
ё
2bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/SelectSelect8bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqual8bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Select/EnterEbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1*(
_output_shapes
:         ђ*
T0*X
_classN
LJloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1
З
8bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Select/EnterEnter+bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/zeros*
T0*X
_classN
LJloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*(
_output_shapes
:         ђ*
parallel_iterations 
ђ
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Select_1Select8bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqual6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_3Cbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value*V
_classL
JHloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value*(
_output_shapes
:         ђ *
T0
ё
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Select_2Select8bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqual6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_4Ebilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1*X
_classN
LJloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1*(
_output_shapes
:         ђ*
T0
█
Obilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Ubilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_12bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Select6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_2*
T0*X
_classN
LJloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1*
_output_shapes
: 
Ѕ
Ubilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArray*
T0*X
_classN
LJloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
_output_shapes
:*
is_constant(*
parallel_iterations 
г
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/add_1/yConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╚
1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/add_1AddV26bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_13bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/add_1/y*
T0*
_output_shapes
: 
ю
9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/NextIterationNextIteration/bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/add*
T0*
_output_shapes
: 
а
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_1NextIteration1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/add_1*
T0*
_output_shapes
: 
Й
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_2NextIterationObilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
х
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_3NextIteration4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Select_1*(
_output_shapes
:         ђ *
T0
х
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_4NextIteration4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Select_2*(
_output_shapes
:         ђ*
T0
Ї
0bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/ExitExit2bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch*
_output_shapes
: *
T0
Љ
2bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Exit_1Exit4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_1*
_output_shapes
: *
T0
Љ
2bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Exit_2Exit4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_2*
_output_shapes
: *
T0
Б
2bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Exit_3Exit4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_3*
T0*(
_output_shapes
:         ђ 
Б
2bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Exit_4Exit4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_4*
T0*(
_output_shapes
:         ђ
╩
Bbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayStack/range/startConst*
value	B : *D
_class:
86loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArray*
dtype0*
_output_shapes
: 
╩
Bbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayStack/range/deltaConst*
value	B :*D
_class:
86loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArray*
dtype0*
_output_shapes
: 
№
<bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayStack/rangeRangeBbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayStack/range/start5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1Bbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayStack/range/delta*D
_class:
86loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArray*#
_output_shapes
:         
и
Jbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV31bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArray<bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayStack/range2bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Exit_2*%
element_shape:         ђ*D
_class:
86loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArray*
dtype0*5
_output_shapes#
!:                  ђ
x
-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Const_4Const*
valueB:ђ*
dtype0*
_output_shapes
:
n
,bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
u
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
u
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ж
-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/range_1Range3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/range_1/start,bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Rank_13bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/range_1/delta*
_output_shapes
:
ѕ
7bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
u
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ё
.bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/concat_2ConcatV27bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/concat_2/values_0-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/range_13bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/concat_2/axis*
_output_shapes
:*
N*
T0
Щ
1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/transpose_1	TransposeJbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayStack/TensorArrayGatherV3.bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/concat_2*
T0*5
_output_shapes#
!:                  ђ
o
bilm/strided_slice_3/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
q
bilm/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*!
valueB"            
q
bilm/strided_slice_3/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
ъ
bilm/strided_slice_3StridedSlice1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/transpose_1bilm/strided_slice_3/stackbilm/strided_slice_3/stack_1bilm/strided_slice_3/stack_2*
Index0*
T0*
end_mask*5
_output_shapes#
!:                  ђ*

begin_mask
i
'bilm/RNN_0/RNN/MultiRNNCell/Cell1/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
і
%bilm/RNN_0/RNN/MultiRNNCell/Cell1/addAddV2Sum'bilm/RNN_0/RNN/MultiRNNCell/Cell1/add/y*
T0*#
_output_shapes
:         
l
*bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
s
1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
s
1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Р
+bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/rangeRange1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/range/start*bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Rank1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/range/delta*
_output_shapes
:
є
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
s
1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
§
,bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/concatConcatV25bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/concat/values_0+bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/range1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/concat/axis*
N*
T0*
_output_shapes
:
П
/bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/transpose	Transpose1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/transpose_1,bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/concat*5
_output_shapes#
!:                  ђ*
T0
ќ
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/sequence_lengthIdentity%bilm/RNN_0/RNN/MultiRNNCell/Cell1/add*
T0*#
_output_shapes
:         
і
+bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ShapeShape/bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/transpose*
T0*
_output_shapes
:
Ѓ
9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
Ё
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ё
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
в
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_sliceStridedSlice+bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Shape9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice/stack;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice/stack_1;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice/stack_2*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0
А
_bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
ц
[bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims
ExpandDims3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims/dim*
T0*
_output_shapes
:
А
Vbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ConstConst*
dtype0*
_output_shapes
:*
valueB:ђ 
ъ
\bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ц
Wbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/concatConcatV2[bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDimsVbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/Const\bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/concat/axis*
N*
T0*
_output_shapes
:
А
\bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╚
Vbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/zerosFillWbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/concat\bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/zeros/Const*(
_output_shapes
:         ђ *
T0
Б
abilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
е
]bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_1
ExpandDims3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_sliceabilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dim*
_output_shapes
:*
T0
Б
Xbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/Const_1Const*
dtype0*
_output_shapes
:*
valueB:ђ 
Б
abilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dimConst*
dtype0*
_output_shapes
: *
value	B : 
е
]bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_2
ExpandDims3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_sliceabilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dim*
_output_shapes
:*
T0
Б
Xbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/Const_2Const*
valueB:ђ*
dtype0*
_output_shapes
:
а
^bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
г
Ybilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/concat_1ConcatV2]bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_2Xbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/Const_2^bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/concat_1/axis*
_output_shapes
:*
N*
T0
Б
^bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
╬
Xbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/zeros_1FillYbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/concat_1^bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/zeros_1/Const*(
_output_shapes
:         ђ*
T0
Б
abilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
е
]bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_3
ExpandDims3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_sliceabilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dim*
_output_shapes
:*
T0
Б
Xbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/Const_3Const*
valueB:ђ*
dtype0*
_output_shapes
:
њ
-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Shape_1Shape5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/sequence_length*
_output_shapes
:*
T0
ќ
+bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/stackPack3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice*
_output_shapes
:*
N*
T0
х
+bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/EqualEqual-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Shape_1+bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/stack*
_output_shapes
:*
T0
u
+bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ConstConst*
valueB: *
dtype0*
_output_shapes
:
б
)bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/AllAll+bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Equal+bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Const*
_output_shapes
: 
╚
2bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Assert/ConstConst*f
value]B[ BUExpected shape for Tensor bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
Ё
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
л
:bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Assert/Assert/data_0Const*f
value]B[ BUExpected shape for Tensor bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
І
:bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Assert/Assert/data_2Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 
╦
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Assert/AssertAssert)bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/All:bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Assert/Assert/data_0+bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/stack:bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Assert/Assert/data_2-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Shape_1*
T
2
п
1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/CheckSeqLenIdentity5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/sequence_length4^bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Assert/Assert*
T0*#
_output_shapes
:         
ї
-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Shape_2Shape/bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/transpose*
T0*
_output_shapes
:
Ё
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
Є
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Є
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ш
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1StridedSlice-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Shape_2;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1/stack=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1/stack_1=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1/stack_2*
T0*
_output_shapes
: *
Index0*
shrink_axis_mask
ї
-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Shape_3Shape/bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/transpose*
T0*
_output_shapes
:
Ё
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
Є
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Є
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ш
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_2StridedSlice-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Shape_3;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_2/stack=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_2/stack_1=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_2/stack_2*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: 
v
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
л
0bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ExpandDims
ExpandDims5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_24bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ExpandDims/dim*
T0*
_output_shapes
:
x
-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Const_1Const*
valueB:ђ*
dtype0*
_output_shapes
:
u
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
■
.bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/concat_1ConcatV20bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ExpandDims-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Const_13bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/concat_1/axis*
N*
T0*
_output_shapes
:
v
1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╔
+bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/zerosFill.bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/concat_11bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/zeros/Const*
T0*(
_output_shapes
:         ђ
w
-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
│
)bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/MinMin1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/CheckSeqLen-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Const_2*
T0*
_output_shapes
: 
w
-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
│
)bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/MaxMax1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/CheckSeqLen-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Const_3*
T0*
_output_shapes
: 
l
*bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
Й
1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayTensorArrayV35bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1*
dtype0*
_output_shapes

:: *Q
tensor_array_name<:bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/dynamic_rnn/output_0*%
element_shape:         ђ*
identical_element_shapes(
┐
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArray_1TensorArrayV35bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1*P
tensor_array_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/dynamic_rnn/input_0*%
element_shape:         ђ*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
Ю
>bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/ShapeShape/bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/transpose*
_output_shapes
:*
T0
ќ
Lbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
ў
Nbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
ў
Nbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╩
Fbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/strided_sliceStridedSlice>bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/ShapeLbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/strided_slice/stackNbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/strided_slice/stack_1Nbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/strided_slice/stack_2*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0
є
Dbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
є
Dbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
└
>bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/rangeRangeDbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/range/startFbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/strided_sliceDbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:         
║
`bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV33bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArray_1>bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/range/bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/transpose5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArray_1:1*
T0*B
_class8
64loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/transpose*
_output_shapes
: 
q
/bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
х
-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/MaximumMaximum/bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Maximum/x)bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Max*
T0*
_output_shapes
: 
┐
-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/MinimumMinimum5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Maximum*
T0*
_output_shapes
: 

=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
ђ
1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/EnterEnter=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/iteration_counter*
parallel_iterations *
T0*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
_output_shapes
: 
№
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter_1Enter*bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/time*
T0*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
_output_shapes
: *
parallel_iterations 
Э
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter_2Enter3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArray:1*
T0*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
_output_shapes
: *
parallel_iterations 
Г
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter_3EnterVbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/zeros*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/while_context*(
_output_shapes
:         ђ *
parallel_iterations *
T0
»
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter_4EnterXbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/zeros_1*
T0*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/while_context*(
_output_shapes
:         ђ*
parallel_iterations 
н
1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/MergeMerge1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration*
N*
T0*
_output_shapes
: : 
┌
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_1Merge3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter_1;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_1*
_output_shapes
: : *
N*
T0
┌
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_2Merge3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter_2;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_2*
N*
T0*
_output_shapes
: : 
В
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_3Merge3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter_3;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_3**
_output_shapes
:         ђ : *
N*
T0
В
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_4Merge3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter_4;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_4*
N*
T0**
_output_shapes
:         ђ: 
─
0bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/LessLess1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Less/Enter*
T0*
_output_shapes
: 
љ
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Less/EnterEnter5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1*
T0*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(*
_output_shapes
: *
parallel_iterations 
╩
2bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Less_1Less3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_18bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Less_1/Enter*
_output_shapes
: *
T0
і
8bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Less_1/EnterEnter-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Minimum*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(*
_output_shapes
: *
parallel_iterations *
T0
┬
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/LogicalAnd
LogicalAnd0bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Less2bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Less_1*
_output_shapes
: 
љ
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/LoopCondLoopCond6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/LogicalAnd*
_output_shapes
: 
ј
2bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/SwitchSwitch1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/LoopCond*
T0*D
_class:
86loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge*
_output_shapes
: : 
ћ
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_1Switch3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_14bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/LoopCond*F
_class<
:8loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_1*
_output_shapes
: : *
T0
ћ
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_2Switch3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_24bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/LoopCond*
_output_shapes
: : *
T0*F
_class<
:8loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_2
И
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_3Switch3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_34bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/LoopCond*F
_class<
:8loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_3*<
_output_shapes*
(:         ђ :         ђ *
T0
И
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_4Switch3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_44bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/LoopCond*F
_class<
:8loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_4*<
_output_shapes*
(:         ђ:         ђ*
T0
Ќ
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/IdentityIdentity4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch:1*
_output_shapes
: *
T0
Џ
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_1Identity6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_1:1*
_output_shapes
: *
T0
Џ
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_2Identity6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
Г
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_3Identity6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_3:1*(
_output_shapes
:         ђ *
T0
Г
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_4Identity6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_4:1*(
_output_shapes
:         ђ*
T0
ф
1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add/yConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
┬
/bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/addAddV24bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add/y*
_output_shapes
: *
T0
═
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3TensorArrayReadV3Cbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3/Enter6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_1Ebilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:         ђ
Ъ
Cbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3/EnterEnter3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArray_1*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(*
_output_shapes
:*
parallel_iterations *
T0
╩
Ebilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3/Enter_1Enter`bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(*
_output_shapes
: *
parallel_iterations 
Ь
8bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqualGreaterEqual6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_1>bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqual/Enter*#
_output_shapes
:         *
T0
А
>bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqual/EnterEnter1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/CheckSeqLen*
parallel_iterations *
T0*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(*#
_output_shapes
:         
з
Wbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"    @  *I
_class?
=;loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*
dtype0*
_output_shapes
:
т
Ubilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *0ў╝*I
_class?
=;loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*
dtype0*
_output_shapes
: 
т
Ubilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *0ў<*I
_class?
=;loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel
═
_bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformWbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/shape*
T0*I
_class?
=;loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*
dtype0*!
_output_shapes
:ђђђ
Ш
Ubilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/subSubUbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/maxUbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
T0*I
_class?
=;loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*
_output_shapes
: 
І
Ubilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/mulMul_bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformUbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/sub*!
_output_shapes
:ђђђ*
T0*I
_class?
=;loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel
§
Qbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniformAddUbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/mulUbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/min*!
_output_shapes
:ђђђ*
T0*I
_class?
=;loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel
ќ
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernelVarHandleOp*
shape:ђђђ*I
_class?
=;loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*
dtype0*
_output_shapes
: *G
shared_name86bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel
й
Wbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*
_output_shapes
: 
ж
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/AssignAssignVariableOp6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernelQbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform*
dtype0
─
Jbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*
dtype0*!
_output_shapes
:ђђђ
╚
Dbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Read/IdentityIdentityJbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Read/ReadVariableOp*
T0*!
_output_shapes
:ђђђ
в
Vbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensorConst*
valueB:ђђ*G
_class=
;9loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias*
dtype0*
_output_shapes
:
┌
Lbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Initializer/zeros/ConstConst*
valueB
 *    *G
_class=
;9loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias*
dtype0*
_output_shapes
: 
С
Fbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Initializer/zerosFillVbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensorLbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Initializer/zeros/Const*
T0*G
_class=
;9loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias*
_output_shapes

:ђђ
І
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/biasVarHandleOp*
shape:ђђ*G
_class=
;9loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias*
dtype0*
_output_shapes
: *E
shared_name64bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias
╣
Ubilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias*
_output_shapes
: 
┌
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/AssignAssignVariableOp4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/biasFbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Initializer/zeros*
dtype0
╗
Hbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Read/ReadVariableOpReadVariableOp4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias*
dtype0*
_output_shapes

:ђђ
┐
Bbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Read/IdentityIdentityHbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Read/ReadVariableOp*
_output_shapes

:ђђ*
T0
Ѕ
bbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/shapeConst*
valueB"      *T
_classJ
HFloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel*
dtype0*
_output_shapes
:
ч
`bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/minConst*
valueB
 *:═й*T
_classJ
HFloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel*
dtype0*
_output_shapes
: 
ч
`bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/maxConst*
valueB
 *:═=*T
_classJ
HFloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel*
dtype0*
_output_shapes
: 
ь
jbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/RandomUniformRandomUniformbbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/shape*
T0*T
_classJ
HFloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel*
dtype0* 
_output_shapes
:
ђ ђ
б
`bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/subSub`bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/max`bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*T
_classJ
HFloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel
Х
`bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/mulMuljbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/RandomUniform`bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/sub*
T0*T
_classJ
HFloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel* 
_output_shapes
:
ђ ђ
е
\bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniformAdd`bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/mul`bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/min* 
_output_shapes
:
ђ ђ*
T0*T
_classJ
HFloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel
Х
Abilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernelVarHandleOp*
shape:
ђ ђ*T
_classJ
HFloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel*
dtype0*R
shared_nameCAbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel*
_output_shapes
: 
М
bbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpAbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel*
_output_shapes
: 
і
Hbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/AssignAssignVariableOpAbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel\bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform*
dtype0
┘
Ubilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Read/ReadVariableOpReadVariableOpAbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel*
dtype0* 
_output_shapes
:
ђ ђ
П
Obilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Read/IdentityIdentityUbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Read/ReadVariableOp* 
_output_shapes
:
ђ ђ*
T0
║
Abilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/concat/axisConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
Й
<bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/concatConcatV2=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV36bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_4Abilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/concat/axis*(
_output_shapes
:         ђ*
N*
T0
Ч
<bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMulMatMul<bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/concatBbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul/Enter*)
_output_shapes
:         ђђ*
T0
Х
Bbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul/EnterEnterDbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Read/Identity*
T0*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(*!
_output_shapes
:ђђђ*
parallel_iterations 
 
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/BiasAddBiasAdd<bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMulCbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/BiasAdd/Enter*)
_output_shapes
:         ђђ*
T0
░
Cbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/BiasAdd/EnterEnterBbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Read/Identity*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(*
_output_shapes

:ђђ*
parallel_iterations *
T0
┤
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/ConstConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Й
Ebilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split/split_dimConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
╩
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/splitSplitEbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split/split_dim=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/BiasAdd*
T0*
	num_split*d
_output_shapesR
P:         ђ :         ђ :         ђ :         ђ 
и
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/add/yConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ы
9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/addAddV2=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split:2;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/add/y*
T0*(
_output_shapes
:         ђ 
Х
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/SigmoidSigmoid9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/add*
T0*(
_output_shapes
:         ђ 
Ж
9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mulMul=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Sigmoid6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_3*(
_output_shapes
:         ђ *
T0
║
?bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Sigmoid_1Sigmoid;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split*
T0*(
_output_shapes
:         ђ 
┤
:bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/TanhTanh=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split:1*(
_output_shapes
:         ђ *
T0
Ы
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mul_1Mul?bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Sigmoid_1:bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Tanh*(
_output_shapes
:         ђ *
T0
№
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/add_1AddV29bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mul;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mul_1*
T0*(
_output_shapes
:         ђ 
╔
Mbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/Minimum/yConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
valueB
 *  @@*
dtype0*
_output_shapes
: 
Ћ
Kbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/MinimumMinimum;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/add_1Mbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/Minimum/y*(
_output_shapes
:         ђ *
T0
┴
Ebilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/yConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  @└
Ћ
Cbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_valueMaximumKbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/MinimumEbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/y*(
_output_shapes
:         ђ *
T0
╝
?bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Sigmoid_2Sigmoid=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split:3*
T0*(
_output_shapes
:         ђ 
╝
<bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Tanh_1TanhCbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value*
T0*(
_output_shapes
:         ђ 
З
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mul_2Mul?bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Sigmoid_2<bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Tanh_1*(
_output_shapes
:         ђ *
T0
■
>bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul_1MatMul;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mul_2Dbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul_1/Enter*(
_output_shapes
:         ђ*
T0
┬
Dbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul_1/EnterEnterObilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Read/Identity*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(* 
_output_shapes
:
ђ ђ*
parallel_iterations *
T0
╦
Obilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/Minimum/yConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
valueB
 *  @@*
dtype0*
_output_shapes
: 
ю
Mbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/MinimumMinimum>bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul_1Obilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/Minimum/y*
T0*(
_output_shapes
:         ђ
├
Gbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/yConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
valueB
 *  @└*
dtype0*
_output_shapes
: 
Џ
Ebilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1MaximumMbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/MinimumGbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/y*
T0*(
_output_shapes
:         ђ
з
1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add_1AddV2=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3Ebilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1*
T0*(
_output_shapes
:         ђ
▄
2bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/SelectSelect8bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqual8bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Select/Enter1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add_1*D
_class:
86loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add_1*(
_output_shapes
:         ђ*
T0
Я
8bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Select/EnterEnter+bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/zeros*
T0*D
_class:
86loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add_1*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(*(
_output_shapes
:         ђ*
parallel_iterations 
ђ
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Select_1Select8bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqual6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_3Cbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value*V
_classL
JHloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value*(
_output_shapes
:         ђ *
T0
ё
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Select_2Select8bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqual6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_4Ebilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1*
T0*X
_classN
LJloc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1*(
_output_shapes
:         ђ
К
Obilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Ubilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_12bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Select6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_2*D
_class:
86loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add_1*
_output_shapes
: *
T0
ш
Ubilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArray*
parallel_iterations *
T0*D
_class:
86loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add_1*I

frame_name;9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
_output_shapes
:*
is_constant(
г
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add_2/yConst5^bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╚
1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add_2AddV26bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_13bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add_2/y*
_output_shapes
: *
T0
ю
9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/NextIterationNextIteration/bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add*
_output_shapes
: *
T0
а
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_1NextIteration1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add_2*
T0*
_output_shapes
: 
Й
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_2NextIterationObilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
х
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_3NextIteration4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Select_1*
T0*(
_output_shapes
:         ђ 
х
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_4NextIteration4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Select_2*
T0*(
_output_shapes
:         ђ
Ї
0bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/ExitExit2bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch*
_output_shapes
: *
T0
Љ
2bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Exit_1Exit4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_1*
T0*
_output_shapes
: 
Љ
2bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Exit_2Exit4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_2*
_output_shapes
: *
T0
Б
2bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Exit_3Exit4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_3*
T0*(
_output_shapes
:         ђ 
Б
2bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Exit_4Exit4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_4*
T0*(
_output_shapes
:         ђ
╩
Bbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayStack/range/startConst*
value	B : *D
_class:
86loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArray*
dtype0*
_output_shapes
: 
╩
Bbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayStack/range/deltaConst*
value	B :*D
_class:
86loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArray*
dtype0*
_output_shapes
: 
№
<bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayStack/rangeRangeBbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayStack/range/start5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1Bbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayStack/range/delta*#
_output_shapes
:         *D
_class:
86loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArray
и
Jbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV31bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArray<bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayStack/range2bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Exit_2*%
element_shape:         ђ*D
_class:
86loc:@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArray*
dtype0*5
_output_shapes#
!:                  ђ
x
-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Const_4Const*
valueB:ђ*
dtype0*
_output_shapes
:
n
,bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
u
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/range_1/startConst*
dtype0*
_output_shapes
: *
value	B :
u
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/range_1/deltaConst*
dtype0*
_output_shapes
: *
value	B :
Ж
-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/range_1Range3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/range_1/start,bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Rank_13bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/range_1/delta*
_output_shapes
:
ѕ
7bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/concat_2/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
u
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ё
.bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/concat_2ConcatV27bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/concat_2/values_0-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/range_13bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/concat_2/axis*
N*
T0*
_output_shapes
:
Щ
1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/transpose_1	TransposeJbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayStack/TensorArrayGatherV3.bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/concat_2*5
_output_shapes#
!:                  ђ*
T0
o
bilm/strided_slice_4/stackConst*
dtype0*
_output_shapes
:*!
valueB"           
q
bilm/strided_slice_4/stack_1Const*
dtype0*
_output_shapes
:*!
valueB"            
q
bilm/strided_slice_4/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
ъ
bilm/strided_slice_4StridedSlice1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/transpose_1bilm/strided_slice_4/stackbilm/strided_slice_4/stack_1bilm/strided_slice_4/stack_2*
T0*
end_mask*5
_output_shapes#
!:                  ђ*

begin_mask*
Index0
Ќ
bilm/ReverseSequenceReverseSequencebilm/Reshape_1Sum*
seq_dim*5
_output_shapes#
!:                  ђ*
T0*

Tlen0
T
bilm/concat_4/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Ў
bilm/concat_4ConcatV2bilm/Tile_1bilm/ReverseSequencebilm/concat_4/axis*
N*
T0*5
_output_shapes#
!:                  ђ
i
'bilm/RNN_1/RNN/MultiRNNCell/Cell0/add/yConst*
dtype0*
_output_shapes
: *
value	B :
і
%bilm/RNN_1/RNN/MultiRNNCell/Cell0/addAddV2Sum'bilm/RNN_1/RNN/MultiRNNCell/Cell0/add/y*
T0*#
_output_shapes
:         
l
*bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
s
1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
s
1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Р
+bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/rangeRange1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/range/start*bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Rank1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/range/delta*
_output_shapes
:
є
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
s
1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
§
,bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/concatConcatV25bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/concat/values_0+bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/range1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/concat/axis*
N*
T0*
_output_shapes
:
╣
/bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/transpose	Transposebilm/concat_4,bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/concat*
T0*5
_output_shapes#
!:                  ђ
ќ
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/sequence_lengthIdentity%bilm/RNN_1/RNN/MultiRNNCell/Cell0/add*#
_output_shapes
:         *
T0
і
+bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/ShapeShape/bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/transpose*
_output_shapes
:*
T0
Ѓ
9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
Ё
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Ё
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
в
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_sliceStridedSlice+bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Shape9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice/stack;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice/stack_1;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice/stack_2*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: 
ѕ
Fbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ы
Bbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims
ExpandDims3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_sliceFbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims/dim*
T0*
_output_shapes
:
ѕ
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ConstConst*
valueB:ђ *
dtype0*
_output_shapes
:
Ё
Cbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
└
>bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/concatConcatV2Bbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ConstCbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/concat/axis*
N*
T0*
_output_shapes
:
ѕ
Cbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
§
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/zerosFill>bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/concatCbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/zeros/Const*(
_output_shapes
:         ђ *
T0
і
Hbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ш
Dbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_1
ExpandDims3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_sliceHbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_1/dim*
T0*
_output_shapes
:
і
?bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/Const_1Const*
dtype0*
_output_shapes
:*
valueB:ђ 
і
Hbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ш
Dbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_2
ExpandDims3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_sliceHbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_2/dim*
T0*
_output_shapes
:
і
?bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/Const_2Const*
valueB:ђ*
dtype0*
_output_shapes
:
Є
Ebilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╚
@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/concat_1ConcatV2Dbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_2?bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/Const_2Ebilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/concat_1/axis*
N*
T0*
_output_shapes
:
і
Ebilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ѓ
?bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/zeros_1Fill@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/concat_1Ebilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/zeros_1/Const*(
_output_shapes
:         ђ*
T0
і
Hbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ш
Dbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_3
ExpandDims3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_sliceHbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/ExpandDims_3/dim*
T0*
_output_shapes
:
і
?bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/Const_3Const*
valueB:ђ*
dtype0*
_output_shapes
:
њ
-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Shape_1Shape5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/sequence_length*
T0*
_output_shapes
:
ќ
+bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/stackPack3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice*
_output_shapes
:*
N*
T0
х
+bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/EqualEqual-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Shape_1+bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/stack*
_output_shapes
:*
T0
u
+bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/ConstConst*
valueB: *
dtype0*
_output_shapes
:
б
)bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/AllAll+bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Equal+bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Const*
_output_shapes
: 
╚
2bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Assert/ConstConst*
dtype0*
_output_shapes
: *f
value]B[ BUExpected shape for Tensor bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/sequence_length:0 is 
Ё
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
л
:bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Assert/Assert/data_0Const*f
value]B[ BUExpected shape for Tensor bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
І
:bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
╦
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Assert/AssertAssert)bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/All:bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Assert/Assert/data_0+bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/stack:bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Assert/Assert/data_2-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Shape_1*
T
2
п
1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/CheckSeqLenIdentity5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/sequence_length4^bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Assert/Assert*
T0*#
_output_shapes
:         
ї
-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Shape_2Shape/bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/transpose*
T0*
_output_shapes
:
Ё
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
Є
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Є
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ш
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1StridedSlice-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Shape_2;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1/stack=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1/stack_1=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1/stack_2*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: 
ї
-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Shape_3Shape/bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/transpose*
_output_shapes
:*
T0
Ё
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
Є
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Є
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ш
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_2StridedSlice-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Shape_3;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_2/stack=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_2/stack_1=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_2/stack_2*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0
v
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
л
0bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/ExpandDims
ExpandDims5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_24bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/ExpandDims/dim*
_output_shapes
:*
T0
x
-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Const_1Const*
valueB:ђ*
dtype0*
_output_shapes
:
u
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
■
.bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/concat_1ConcatV20bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/ExpandDims-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Const_13bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/concat_1/axis*
_output_shapes
:*
N*
T0
v
1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╔
+bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/zerosFill.bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/concat_11bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/zeros/Const*
T0*(
_output_shapes
:         ђ
w
-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
│
)bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/MinMin1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/CheckSeqLen-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Const_2*
_output_shapes
: *
T0
w
-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
│
)bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/MaxMax1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/CheckSeqLen-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Const_3*
T0*
_output_shapes
: 
l
*bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/timeConst*
dtype0*
_output_shapes
: *
value	B : 
Й
1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayTensorArrayV35bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1*Q
tensor_array_name<:bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/dynamic_rnn/output_0*%
element_shape:         ђ*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
┐
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArray_1TensorArrayV35bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1*
dtype0*
_output_shapes

:: *P
tensor_array_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/dynamic_rnn/input_0*%
element_shape:         ђ*
identical_element_shapes(
Ю
>bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/ShapeShape/bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/transpose*
_output_shapes
:*
T0
ќ
Lbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
ў
Nbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
ў
Nbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╩
Fbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/strided_sliceStridedSlice>bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/ShapeLbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/strided_slice/stackNbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/strided_slice/stack_1Nbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/strided_slice/stack_2*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: 
є
Dbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
є
Dbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
└
>bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/rangeRangeDbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/range/startFbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/strided_sliceDbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:         
║
`bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV33bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArray_1>bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/range/bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/transpose5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArray_1:1*
_output_shapes
: *
T0*B
_class8
64loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/transpose
q
/bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
х
-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/MaximumMaximum/bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Maximum/x)bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Max*
T0*
_output_shapes
: 
┐
-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/MinimumMinimum5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Maximum*
T0*
_output_shapes
: 

=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/iteration_counterConst*
dtype0*
_output_shapes
: *
value	B : 
ђ
1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/EnterEnter=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/iteration_counter*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
_output_shapes
: *
parallel_iterations *
T0
№
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter_1Enter*bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/time*
T0*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
_output_shapes
: *
parallel_iterations 
Э
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter_2Enter3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArray:1*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
_output_shapes
: *
parallel_iterations *
T0
ћ
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter_3Enter=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/zeros*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/while_context*(
_output_shapes
:         ђ *
parallel_iterations *
T0
ќ
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter_4Enter?bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/LSTMCellZeroState/zeros_1*
T0*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/while_context*(
_output_shapes
:         ђ*
parallel_iterations 
н
1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/MergeMerge1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration*
_output_shapes
: : *
N*
T0
┌
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_1Merge3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter_1;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_1*
N*
T0*
_output_shapes
: : 
┌
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_2Merge3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter_2;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_2*
N*
T0*
_output_shapes
: : 
В
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_3Merge3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter_3;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_3**
_output_shapes
:         ђ : *
N*
T0
В
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_4Merge3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter_4;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_4*
N*
T0**
_output_shapes
:         ђ: 
─
0bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/LessLess1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Less/Enter*
T0*
_output_shapes
: 
љ
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Less/EnterEnter5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*
_output_shapes
: *
parallel_iterations *
T0
╩
2bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Less_1Less3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_18bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
і
8bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Less_1/EnterEnter-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Minimum*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*
_output_shapes
: *
parallel_iterations *
T0
┬
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/LogicalAnd
LogicalAnd0bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Less2bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Less_1*
_output_shapes
: 
љ
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/LoopCondLoopCond6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/LogicalAnd*
_output_shapes
: 
ј
2bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/SwitchSwitch1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/LoopCond*
T0*D
_class:
86loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge*
_output_shapes
: : 
ћ
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_1Switch3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_14bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/LoopCond*
T0*F
_class<
:8loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_1*
_output_shapes
: : 
ћ
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_2Switch3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_24bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/LoopCond*F
_class<
:8loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_2*
_output_shapes
: : *
T0
И
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_3Switch3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_34bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/LoopCond*F
_class<
:8loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_3*<
_output_shapes*
(:         ђ :         ђ *
T0
И
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_4Switch3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_44bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/LoopCond*F
_class<
:8loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_4*<
_output_shapes*
(:         ђ:         ђ*
T0
Ќ
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/IdentityIdentity4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch:1*
_output_shapes
: *
T0
Џ
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_1Identity6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
Џ
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_2Identity6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_2:1*
_output_shapes
: *
T0
Г
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_3Identity6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_3:1*
T0*(
_output_shapes
:         ђ 
Г
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_4Identity6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_4:1*(
_output_shapes
:         ђ*
T0
ф
1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/add/yConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
┬
/bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/addAddV24bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/add/y*
_output_shapes
: *
T0
═
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3TensorArrayReadV3Cbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3/Enter6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_1Ebilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:         ђ
Ъ
Cbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3/EnterEnter3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArray_1*
parallel_iterations *
T0*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*
_output_shapes
:
╩
Ebilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3/Enter_1Enter`bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*
_output_shapes
: *
parallel_iterations 
Ь
8bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqualGreaterEqual6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_1>bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqual/Enter*
T0*#
_output_shapes
:         
А
>bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqual/EnterEnter1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/CheckSeqLen*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*#
_output_shapes
:         *
parallel_iterations *
T0
з
Wbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"    @  *I
_class?
=;loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*
dtype0*
_output_shapes
:
т
Ubilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *0ў╝*I
_class?
=;loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*
dtype0*
_output_shapes
: 
т
Ubilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *0ў<*I
_class?
=;loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*
dtype0*
_output_shapes
: 
═
_bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformWbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/shape*
T0*I
_class?
=;loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*
dtype0*!
_output_shapes
:ђђђ
Ш
Ubilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/subSubUbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/maxUbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
T0*I
_class?
=;loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*
_output_shapes
: 
І
Ubilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/mulMul_bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformUbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/sub*I
_class?
=;loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*!
_output_shapes
:ђђђ*
T0
§
Qbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniformAddUbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/mulUbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform/min*!
_output_shapes
:ђђђ*
T0*I
_class?
=;loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel
ќ
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernelVarHandleOp*
shape:ђђђ*I
_class?
=;loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*
dtype0*G
shared_name86bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*
_output_shapes
: 
й
Wbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*
_output_shapes
: 
ж
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/AssignAssignVariableOp6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernelQbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform*
dtype0
─
Jbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel*
dtype0*!
_output_shapes
:ђђђ
╚
Dbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Read/IdentityIdentityJbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Read/ReadVariableOp*
T0*!
_output_shapes
:ђђђ
в
Vbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensorConst*
valueB:ђђ*G
_class=
;9loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias*
dtype0*
_output_shapes
:
┌
Lbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Initializer/zeros/ConstConst*
valueB
 *    *G
_class=
;9loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias*
dtype0*
_output_shapes
: 
С
Fbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Initializer/zerosFillVbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensorLbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Initializer/zeros/Const*G
_class=
;9loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias*
_output_shapes

:ђђ*
T0
І
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/biasVarHandleOp*
shape:ђђ*G
_class=
;9loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias*
dtype0*
_output_shapes
: *E
shared_name64bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias
╣
Ubilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias*
_output_shapes
: 
┌
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/AssignAssignVariableOp4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/biasFbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Initializer/zeros*
dtype0
╗
Hbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Read/ReadVariableOpReadVariableOp4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias*
dtype0*
_output_shapes

:ђђ
┐
Bbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Read/IdentityIdentityHbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Read/ReadVariableOp*
T0*
_output_shapes

:ђђ
Ѕ
bbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/shapeConst*
valueB"      *T
_classJ
HFloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel*
dtype0*
_output_shapes
:
ч
`bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/minConst*
valueB
 *:═й*T
_classJ
HFloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel*
dtype0*
_output_shapes
: 
ч
`bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/maxConst*
valueB
 *:═=*T
_classJ
HFloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel*
dtype0*
_output_shapes
: 
ь
jbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/RandomUniformRandomUniformbbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/shape*
T0*T
_classJ
HFloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel*
dtype0* 
_output_shapes
:
ђ ђ
б
`bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/subSub`bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/max`bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/min*
T0*T
_classJ
HFloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel*
_output_shapes
: 
Х
`bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/mulMuljbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/RandomUniform`bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/sub*
T0*T
_classJ
HFloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel* 
_output_shapes
:
ђ ђ
е
\bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniformAdd`bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/mul`bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/min*
T0*T
_classJ
HFloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel* 
_output_shapes
:
ђ ђ
Х
Abilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernelVarHandleOp*
shape:
ђ ђ*T
_classJ
HFloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel*
dtype0*
_output_shapes
: *R
shared_nameCAbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel
М
bbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpAbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel*
_output_shapes
: 
і
Hbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/AssignAssignVariableOpAbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel\bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform*
dtype0
┘
Ubilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Read/ReadVariableOpReadVariableOpAbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel*
dtype0* 
_output_shapes
:
ђ ђ
П
Obilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Read/IdentityIdentityUbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Read/ReadVariableOp* 
_output_shapes
:
ђ ђ*
T0
║
Abilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/concat/axisConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Й
<bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/concatConcatV2=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV36bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_4Abilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/concat/axis*(
_output_shapes
:         ђ*
N*
T0
Ч
<bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMulMatMul<bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/concatBbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul/Enter*)
_output_shapes
:         ђђ*
T0
Х
Bbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul/EnterEnterDbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Read/Identity*
T0*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*!
_output_shapes
:ђђђ*
parallel_iterations 
 
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/BiasAddBiasAdd<bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMulCbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/BiasAdd/Enter*
T0*)
_output_shapes
:         ђђ
░
Cbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/BiasAdd/EnterEnterBbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Read/Identity*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*
_output_shapes

:ђђ*
parallel_iterations *
T0
┤
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/ConstConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
Й
Ebilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split/split_dimConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╩
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/splitSplitEbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split/split_dim=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/BiasAdd*d
_output_shapesR
P:         ђ :         ђ :         ђ :         ђ *
T0*
	num_split
и
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/add/yConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
ы
9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/addAddV2=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split:2;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/add/y*
T0*(
_output_shapes
:         ђ 
Х
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/SigmoidSigmoid9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/add*
T0*(
_output_shapes
:         ђ 
Ж
9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mulMul=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Sigmoid6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_3*(
_output_shapes
:         ђ *
T0
║
?bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Sigmoid_1Sigmoid;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split*
T0*(
_output_shapes
:         ђ 
┤
:bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/TanhTanh=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split:1*
T0*(
_output_shapes
:         ђ 
Ы
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mul_1Mul?bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Sigmoid_1:bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Tanh*
T0*(
_output_shapes
:         ђ 
№
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/add_1AddV29bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mul;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mul_1*
T0*(
_output_shapes
:         ђ 
╔
Mbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/Minimum/yConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
valueB
 *  @@*
dtype0*
_output_shapes
: 
Ћ
Kbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/MinimumMinimum;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/add_1Mbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/Minimum/y*(
_output_shapes
:         ђ *
T0
┴
Ebilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/yConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
valueB
 *  @└*
dtype0*
_output_shapes
: 
Ћ
Cbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_valueMaximumKbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/MinimumEbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/y*
T0*(
_output_shapes
:         ђ 
╝
?bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Sigmoid_2Sigmoid=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split:3*(
_output_shapes
:         ђ *
T0
╝
<bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Tanh_1TanhCbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value*
T0*(
_output_shapes
:         ђ 
З
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mul_2Mul?bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Sigmoid_2<bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Tanh_1*(
_output_shapes
:         ђ *
T0
■
>bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul_1MatMul;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mul_2Dbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul_1/Enter*
T0*(
_output_shapes
:         ђ
┬
Dbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul_1/EnterEnterObilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Read/Identity*
T0*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(* 
_output_shapes
:
ђ ђ*
parallel_iterations 
╦
Obilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/Minimum/yConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
valueB
 *  @@*
dtype0*
_output_shapes
: 
ю
Mbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/MinimumMinimum>bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul_1Obilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/Minimum/y*(
_output_shapes
:         ђ*
T0
├
Gbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/yConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  @└
Џ
Ebilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1MaximumMbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/MinimumGbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/y*
T0*(
_output_shapes
:         ђ
ё
2bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/SelectSelect8bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqual8bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Select/EnterEbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1*X
_classN
LJloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1*(
_output_shapes
:         ђ*
T0
З
8bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Select/EnterEnter+bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/zeros*
T0*X
_classN
LJloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*(
_output_shapes
:         ђ*
parallel_iterations 
ђ
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Select_1Select8bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqual6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_3Cbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value*
T0*V
_classL
JHloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value*(
_output_shapes
:         ђ 
ё
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Select_2Select8bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqual6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_4Ebilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1*X
_classN
LJloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1*(
_output_shapes
:         ђ*
T0
█
Obilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Ubilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_12bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Select6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_2*X
_classN
LJloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1*
_output_shapes
: *
T0
Ѕ
Ubilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArray*
T0*X
_classN
LJloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/while_context*
is_constant(*
_output_shapes
:*
parallel_iterations 
г
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/add_1/yConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╚
1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/add_1AddV26bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_13bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/add_1/y*
_output_shapes
: *
T0
ю
9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/NextIterationNextIteration/bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/add*
T0*
_output_shapes
: 
а
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_1NextIteration1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/add_1*
T0*
_output_shapes
: 
Й
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_2NextIterationObilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
х
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_3NextIteration4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Select_1*(
_output_shapes
:         ђ *
T0
х
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_4NextIteration4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Select_2*
T0*(
_output_shapes
:         ђ
Ї
0bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/ExitExit2bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch*
_output_shapes
: *
T0
Љ
2bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Exit_1Exit4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_1*
_output_shapes
: *
T0
Љ
2bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Exit_2Exit4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_2*
T0*
_output_shapes
: 
Б
2bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Exit_3Exit4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_3*(
_output_shapes
:         ђ *
T0
Б
2bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Exit_4Exit4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_4*(
_output_shapes
:         ђ*
T0
╩
Bbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayStack/range/startConst*
dtype0*
_output_shapes
: *
value	B : *D
_class:
86loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArray
╩
Bbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayStack/range/deltaConst*
value	B :*D
_class:
86loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArray*
dtype0*
_output_shapes
: 
№
<bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayStack/rangeRangeBbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayStack/range/start5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1Bbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayStack/range/delta*D
_class:
86loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArray*#
_output_shapes
:         
и
Jbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV31bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArray<bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayStack/range2bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Exit_2*%
element_shape:         ђ*D
_class:
86loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArray*
dtype0*5
_output_shapes#
!:                  ђ
x
-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Const_4Const*
valueB:ђ*
dtype0*
_output_shapes
:
n
,bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
u
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/range_1/startConst*
dtype0*
_output_shapes
: *
value	B :
u
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ж
-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/range_1Range3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/range_1/start,bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Rank_13bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/range_1/delta*
_output_shapes
:
ѕ
7bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
u
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ё
.bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/concat_2ConcatV27bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/concat_2/values_0-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/range_13bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/concat_2/axis*
N*
T0*
_output_shapes
:
Щ
1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/transpose_1	TransposeJbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayStack/TensorArrayGatherV3.bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/concat_2*
T0*5
_output_shapes#
!:                  ђ
o
bilm/strided_slice_5/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
q
bilm/strided_slice_5/stack_1Const*!
valueB"            *
dtype0*
_output_shapes
:
q
bilm/strided_slice_5/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
ъ
bilm/strided_slice_5StridedSlice1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/transpose_1bilm/strided_slice_5/stackbilm/strided_slice_5/stack_1bilm/strided_slice_5/stack_2*
T0*
end_mask*5
_output_shapes#
!:                  ђ*

begin_mask*
Index0
Ъ
bilm/ReverseSequence_1ReverseSequencebilm/strided_slice_5Sum*
seq_dim*5
_output_shapes#
!:                  ђ*
T0*

Tlen0
i
'bilm/RNN_1/RNN/MultiRNNCell/Cell1/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
і
%bilm/RNN_1/RNN/MultiRNNCell/Cell1/addAddV2Sum'bilm/RNN_1/RNN/MultiRNNCell/Cell1/add/y*
T0*#
_output_shapes
:         
l
*bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
s
1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
s
1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Р
+bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/rangeRange1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/range/start*bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Rank1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/range/delta*
_output_shapes
:
є
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB"       
s
1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
§
,bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/concatConcatV25bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/concat/values_0+bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/range1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/concat/axis*
_output_shapes
:*
N*
T0
П
/bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/transpose	Transpose1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/transpose_1,bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/concat*5
_output_shapes#
!:                  ђ*
T0
ќ
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/sequence_lengthIdentity%bilm/RNN_1/RNN/MultiRNNCell/Cell1/add*#
_output_shapes
:         *
T0
і
+bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ShapeShape/bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/transpose*
_output_shapes
:*
T0
Ѓ
9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
Ё
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Ё
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
в
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_sliceStridedSlice+bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Shape9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice/stack;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice/stack_1;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice/stack_2*
T0*
_output_shapes
: *
Index0*
shrink_axis_mask
А
_bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
ц
[bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims
ExpandDims3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims/dim*
T0*
_output_shapes
:
А
Vbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ConstConst*
valueB:ђ *
dtype0*
_output_shapes
:
ъ
\bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ц
Wbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/concatConcatV2[bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDimsVbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/Const\bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/concat/axis*
N*
T0*
_output_shapes
:
А
\bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╚
Vbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/zerosFillWbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/concat\bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/zeros/Const*(
_output_shapes
:         ђ *
T0
Б
abilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
е
]bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_1
ExpandDims3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_sliceabilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dim*
_output_shapes
:*
T0
Б
Xbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/Const_1Const*
valueB:ђ *
dtype0*
_output_shapes
:
Б
abilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
е
]bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_2
ExpandDims3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_sliceabilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dim*
T0*
_output_shapes
:
Б
Xbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/Const_2Const*
valueB:ђ*
dtype0*
_output_shapes
:
а
^bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
г
Ybilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/concat_1ConcatV2]bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_2Xbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/Const_2^bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/concat_1/axis*
N*
T0*
_output_shapes
:
Б
^bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╬
Xbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/zeros_1FillYbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/concat_1^bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/zeros_1/Const*(
_output_shapes
:         ђ*
T0
Б
abilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
е
]bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_3
ExpandDims3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_sliceabilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dim*
_output_shapes
:*
T0
Б
Xbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/Const_3Const*
valueB:ђ*
dtype0*
_output_shapes
:
њ
-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Shape_1Shape5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/sequence_length*
T0*
_output_shapes
:
ќ
+bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/stackPack3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice*
_output_shapes
:*
N*
T0
х
+bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/EqualEqual-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Shape_1+bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/stack*
_output_shapes
:*
T0
u
+bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
б
)bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/AllAll+bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Equal+bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Const*
_output_shapes
: 
╚
2bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Assert/ConstConst*f
value]B[ BUExpected shape for Tensor bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
Ё
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
л
:bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Assert/Assert/data_0Const*f
value]B[ BUExpected shape for Tensor bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
І
:bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Assert/Assert/data_2Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 
╦
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Assert/AssertAssert)bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/All:bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Assert/Assert/data_0+bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/stack:bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Assert/Assert/data_2-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Shape_1*
T
2
п
1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/CheckSeqLenIdentity5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/sequence_length4^bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Assert/Assert*#
_output_shapes
:         *
T0
ї
-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Shape_2Shape/bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/transpose*
T0*
_output_shapes
:
Ё
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
Є
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Є
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ш
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1StridedSlice-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Shape_2;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1/stack=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1/stack_1=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1/stack_2*
T0*
_output_shapes
: *
Index0*
shrink_axis_mask
ї
-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Shape_3Shape/bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/transpose*
T0*
_output_shapes
:
Ё
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
Є
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Є
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ш
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_2StridedSlice-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Shape_3;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_2/stack=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_2/stack_1=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_2/stack_2*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: 
v
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
л
0bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ExpandDims
ExpandDims5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_24bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ExpandDims/dim*
T0*
_output_shapes
:
x
-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Const_1Const*
valueB:ђ*
dtype0*
_output_shapes
:
u
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
■
.bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/concat_1ConcatV20bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ExpandDims-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Const_13bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/concat_1/axis*
N*
T0*
_output_shapes
:
v
1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╔
+bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/zerosFill.bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/concat_11bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/zeros/Const*
T0*(
_output_shapes
:         ђ
w
-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Const_2Const*
dtype0*
_output_shapes
:*
valueB: 
│
)bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/MinMin1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/CheckSeqLen-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Const_2*
T0*
_output_shapes
: 
w
-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Const_3Const*
dtype0*
_output_shapes
:*
valueB: 
│
)bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/MaxMax1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/CheckSeqLen-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Const_3*
_output_shapes
: *
T0
l
*bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
Й
1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayTensorArrayV35bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1*Q
tensor_array_name<:bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/dynamic_rnn/output_0*%
element_shape:         ђ*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
┐
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArray_1TensorArrayV35bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1*P
tensor_array_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/dynamic_rnn/input_0*%
element_shape:         ђ*
identical_element_shapes(*
dtype0*
_output_shapes

:: 
Ю
>bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/ShapeShape/bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/transpose*
T0*
_output_shapes
:
ќ
Lbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
ў
Nbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
ў
Nbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╩
Fbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/strided_sliceStridedSlice>bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/ShapeLbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/strided_slice/stackNbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/strided_slice/stack_1Nbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/strided_slice/stack_2*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0
є
Dbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
є
Dbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
└
>bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/rangeRangeDbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/range/startFbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/strided_sliceDbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:         
║
`bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV33bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArray_1>bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/range/bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/transpose5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArray_1:1*
T0*B
_class8
64loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/transpose*
_output_shapes
: 
q
/bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
х
-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/MaximumMaximum/bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Maximum/x)bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Max*
T0*
_output_shapes
: 
┐
-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/MinimumMinimum5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Maximum*
T0*
_output_shapes
: 

=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/iteration_counterConst*
dtype0*
_output_shapes
: *
value	B : 
ђ
1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/EnterEnter=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/iteration_counter*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
_output_shapes
: *
parallel_iterations *
T0
№
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter_1Enter*bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/time*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
_output_shapes
: *
parallel_iterations *
T0
Э
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter_2Enter3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArray:1*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
_output_shapes
: *
parallel_iterations *
T0
Г
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter_3EnterVbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/zeros*
parallel_iterations *
T0*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/while_context*(
_output_shapes
:         ђ 
»
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter_4EnterXbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/ResidualWrapperZeroState/LSTMCellZeroState/zeros_1*
T0*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/while_context*(
_output_shapes
:         ђ*
parallel_iterations 
н
1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/MergeMerge1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration*
_output_shapes
: : *
N*
T0
┌
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_1Merge3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter_1;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_1*
N*
T0*
_output_shapes
: : 
┌
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_2Merge3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter_2;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_2*
N*
T0*
_output_shapes
: : 
В
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_3Merge3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter_3;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_3**
_output_shapes
:         ђ : *
N*
T0
В
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_4Merge3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter_4;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_4**
_output_shapes
:         ђ: *
N*
T0
─
0bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/LessLess1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Less/Enter*
T0*
_output_shapes
: 
љ
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Less/EnterEnter5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1*
parallel_iterations *
T0*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(*
_output_shapes
: 
╩
2bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Less_1Less3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_18bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
і
8bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Less_1/EnterEnter-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Minimum*
parallel_iterations *
T0*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(*
_output_shapes
: 
┬
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/LogicalAnd
LogicalAnd0bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Less2bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Less_1*
_output_shapes
: 
љ
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/LoopCondLoopCond6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/LogicalAnd*
_output_shapes
: 
ј
2bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/SwitchSwitch1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/LoopCond*D
_class:
86loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge*
_output_shapes
: : *
T0
ћ
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_1Switch3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_14bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/LoopCond*F
_class<
:8loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_1*
_output_shapes
: : *
T0
ћ
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_2Switch3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_24bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/LoopCond*
_output_shapes
: : *
T0*F
_class<
:8loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_2
И
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_3Switch3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_34bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/LoopCond*
T0*F
_class<
:8loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_3*<
_output_shapes*
(:         ђ :         ђ 
И
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_4Switch3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_44bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/LoopCond*
T0*F
_class<
:8loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_4*<
_output_shapes*
(:         ђ:         ђ
Ќ
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/IdentityIdentity4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch:1*
T0*
_output_shapes
: 
Џ
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_1Identity6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_1:1*
_output_shapes
: *
T0
Џ
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_2Identity6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
Г
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_3Identity6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_3:1*
T0*(
_output_shapes
:         ђ 
Г
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_4Identity6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_4:1*(
_output_shapes
:         ђ*
T0
ф
1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add/yConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
┬
/bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/addAddV24bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add/y*
_output_shapes
: *
T0
═
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3TensorArrayReadV3Cbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3/Enter6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_1Ebilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:         ђ
Ъ
Cbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3/EnterEnter3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArray_1*
parallel_iterations *
T0*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(*
_output_shapes
:
╩
Ebilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3/Enter_1Enter`bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
T0*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(*
_output_shapes
: 
Ь
8bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqualGreaterEqual6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_1>bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqual/Enter*#
_output_shapes
:         *
T0
А
>bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqual/EnterEnter1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/CheckSeqLen*
parallel_iterations *
T0*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(*#
_output_shapes
:         
з
Wbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"    @  *I
_class?
=;loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*
dtype0*
_output_shapes
:
т
Ubilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *0ў╝*I
_class?
=;loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*
dtype0*
_output_shapes
: 
т
Ubilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *0ў<*I
_class?
=;loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*
dtype0*
_output_shapes
: 
═
_bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformWbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/shape*
T0*I
_class?
=;loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*
dtype0*!
_output_shapes
:ђђђ
Ш
Ubilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/subSubUbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/maxUbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/min*I
_class?
=;loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*
_output_shapes
: *
T0
І
Ubilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/mulMul_bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformUbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/sub*I
_class?
=;loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*!
_output_shapes
:ђђђ*
T0
§
Qbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniformAddUbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/mulUbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform/min*I
_class?
=;loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*!
_output_shapes
:ђђђ*
T0
ќ
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernelVarHandleOp*
shape:ђђђ*I
_class?
=;loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*
dtype0*
_output_shapes
: *G
shared_name86bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel
й
Wbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*
_output_shapes
: 
ж
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/AssignAssignVariableOp6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernelQbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform*
dtype0
─
Jbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel*
dtype0*!
_output_shapes
:ђђђ
╚
Dbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Read/IdentityIdentityJbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Read/ReadVariableOp*!
_output_shapes
:ђђђ*
T0
в
Vbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensorConst*
valueB:ђђ*G
_class=
;9loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias*
dtype0*
_output_shapes
:
┌
Lbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Initializer/zeros/ConstConst*
valueB
 *    *G
_class=
;9loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias*
dtype0*
_output_shapes
: 
С
Fbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Initializer/zerosFillVbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensorLbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Initializer/zeros/Const*
T0*G
_class=
;9loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias*
_output_shapes

:ђђ
І
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/biasVarHandleOp*
dtype0*
_output_shapes
: *E
shared_name64bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias*
shape:ђђ*G
_class=
;9loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias
╣
Ubilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias*
_output_shapes
: 
┌
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/AssignAssignVariableOp4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/biasFbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Initializer/zeros*
dtype0
╗
Hbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Read/ReadVariableOpReadVariableOp4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias*
dtype0*
_output_shapes

:ђђ
┐
Bbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Read/IdentityIdentityHbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Read/ReadVariableOp*
T0*
_output_shapes

:ђђ
Ѕ
bbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/shapeConst*
valueB"      *T
_classJ
HFloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel*
dtype0*
_output_shapes
:
ч
`bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/minConst*
valueB
 *:═й*T
_classJ
HFloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel*
dtype0*
_output_shapes
: 
ч
`bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/maxConst*
valueB
 *:═=*T
_classJ
HFloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel*
dtype0*
_output_shapes
: 
ь
jbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/RandomUniformRandomUniformbbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
ђ ђ*
T0*T
_classJ
HFloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel
б
`bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/subSub`bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/max`bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/min*T
_classJ
HFloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel*
_output_shapes
: *
T0
Х
`bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/mulMuljbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/RandomUniform`bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
ђ ђ*
T0*T
_classJ
HFloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel
е
\bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniformAdd`bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/mul`bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform/min*T
_classJ
HFloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel* 
_output_shapes
:
ђ ђ*
T0
Х
Abilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernelVarHandleOp*
dtype0*R
shared_nameCAbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel*
_output_shapes
: *
shape:
ђ ђ*T
_classJ
HFloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel
М
bbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpAbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel*
_output_shapes
: 
і
Hbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/AssignAssignVariableOpAbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel\bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform*
dtype0
┘
Ubilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Read/ReadVariableOpReadVariableOpAbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel*
dtype0* 
_output_shapes
:
ђ ђ
П
Obilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Read/IdentityIdentityUbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Read/ReadVariableOp*
T0* 
_output_shapes
:
ђ ђ
║
Abilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/concat/axisConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Й
<bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/concatConcatV2=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV36bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_4Abilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/concat/axis*
N*
T0*(
_output_shapes
:         ђ
Ч
<bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMulMatMul<bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/concatBbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul/Enter*)
_output_shapes
:         ђђ*
T0
Х
Bbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul/EnterEnterDbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Read/Identity*
T0*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(*!
_output_shapes
:ђђђ*
parallel_iterations 
 
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/BiasAddBiasAdd<bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMulCbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/BiasAdd/Enter*)
_output_shapes
:         ђђ*
T0
░
Cbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/BiasAdd/EnterEnterBbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Read/Identity*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(*
_output_shapes

:ђђ*
parallel_iterations *
T0
┤
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/ConstConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Й
Ebilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split/split_dimConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╩
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/splitSplitEbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split/split_dim=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/BiasAdd*
T0*
	num_split*d
_output_shapesR
P:         ђ :         ђ :         ђ :         ђ 
и
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/add/yConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
ы
9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/addAddV2=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split:2;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/add/y*
T0*(
_output_shapes
:         ђ 
Х
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/SigmoidSigmoid9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/add*
T0*(
_output_shapes
:         ђ 
Ж
9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mulMul=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Sigmoid6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_3*
T0*(
_output_shapes
:         ђ 
║
?bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Sigmoid_1Sigmoid;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split*
T0*(
_output_shapes
:         ђ 
┤
:bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/TanhTanh=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split:1*(
_output_shapes
:         ђ *
T0
Ы
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mul_1Mul?bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Sigmoid_1:bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Tanh*(
_output_shapes
:         ђ *
T0
№
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/add_1AddV29bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mul;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mul_1*(
_output_shapes
:         ђ *
T0
╔
Mbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/Minimum/yConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  @@
Ћ
Kbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/MinimumMinimum;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/add_1Mbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/Minimum/y*
T0*(
_output_shapes
:         ђ 
┴
Ebilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/yConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
valueB
 *  @└*
dtype0*
_output_shapes
: 
Ћ
Cbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_valueMaximumKbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/MinimumEbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/y*
T0*(
_output_shapes
:         ђ 
╝
?bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Sigmoid_2Sigmoid=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split:3*(
_output_shapes
:         ђ *
T0
╝
<bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Tanh_1TanhCbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value*(
_output_shapes
:         ђ *
T0
З
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mul_2Mul?bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Sigmoid_2<bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Tanh_1*(
_output_shapes
:         ђ *
T0
■
>bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul_1MatMul;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mul_2Dbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul_1/Enter*
T0*(
_output_shapes
:         ђ
┬
Dbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul_1/EnterEnterObilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Read/Identity*
parallel_iterations *
T0*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
is_constant(* 
_output_shapes
:
ђ ђ
╦
Obilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/Minimum/yConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
valueB
 *  @@*
dtype0*
_output_shapes
: 
ю
Mbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/MinimumMinimum>bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul_1Obilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/Minimum/y*(
_output_shapes
:         ђ*
T0
├
Gbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/yConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  @└
Џ
Ebilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1MaximumMbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/MinimumGbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/y*(
_output_shapes
:         ђ*
T0
з
1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add_1AddV2=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3Ebilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1*
T0*(
_output_shapes
:         ђ
▄
2bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/SelectSelect8bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqual8bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Select/Enter1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add_1*
T0*D
_class:
86loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add_1*(
_output_shapes
:         ђ
Я
8bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Select/EnterEnter+bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/zeros*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/while_context*(
_output_shapes
:         ђ*
is_constant(*
parallel_iterations *
T0*D
_class:
86loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add_1
ђ
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Select_1Select8bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqual6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_3Cbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value*V
_classL
JHloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value*(
_output_shapes
:         ђ *
T0
ё
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Select_2Select8bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqual6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_4Ebilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1*
T0*X
_classN
LJloc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1*(
_output_shapes
:         ђ
К
Obilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Ubilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_12bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Select6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_2*D
_class:
86loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add_1*
_output_shapes
: *
T0
ш
Ubilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArray*D
_class:
86loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add_1*I

frame_name;9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/while_context*
_output_shapes
:*
is_constant(*
parallel_iterations *
T0
г
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add_2/yConst5^bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╚
1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add_2AddV26bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_13bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add_2/y*
T0*
_output_shapes
: 
ю
9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/NextIterationNextIteration/bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add*
T0*
_output_shapes
: 
а
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_1NextIteration1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add_2*
_output_shapes
: *
T0
Й
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_2NextIterationObilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
х
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_3NextIteration4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Select_1*(
_output_shapes
:         ђ *
T0
х
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_4NextIteration4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Select_2*(
_output_shapes
:         ђ*
T0
Ї
0bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/ExitExit2bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch*
_output_shapes
: *
T0
Љ
2bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Exit_1Exit4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_1*
_output_shapes
: *
T0
Љ
2bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Exit_2Exit4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_2*
_output_shapes
: *
T0
Б
2bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Exit_3Exit4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_3*(
_output_shapes
:         ђ *
T0
Б
2bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Exit_4Exit4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_4*
T0*(
_output_shapes
:         ђ
╩
Bbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayStack/range/startConst*
value	B : *D
_class:
86loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArray*
dtype0*
_output_shapes
: 
╩
Bbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayStack/range/deltaConst*
value	B :*D
_class:
86loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArray*
dtype0*
_output_shapes
: 
№
<bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayStack/rangeRangeBbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayStack/range/start5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1Bbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayStack/range/delta*D
_class:
86loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArray*#
_output_shapes
:         
и
Jbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV31bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArray<bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayStack/range2bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Exit_2*%
element_shape:         ђ*D
_class:
86loc:@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArray*
dtype0*5
_output_shapes#
!:                  ђ
x
-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Const_4Const*
dtype0*
_output_shapes
:*
valueB:ђ
n
,bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
u
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
u
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ж
-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/range_1Range3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/range_1/start,bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Rank_13bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/range_1/delta*
_output_shapes
:
ѕ
7bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
u
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ё
.bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/concat_2ConcatV27bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/concat_2/values_0-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/range_13bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/concat_2/axis*
N*
T0*
_output_shapes
:
Щ
1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/transpose_1	TransposeJbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayStack/TensorArrayGatherV3.bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/concat_2*5
_output_shapes#
!:                  ђ*
T0
o
bilm/strided_slice_6/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
q
bilm/strided_slice_6/stack_1Const*
dtype0*
_output_shapes
:*!
valueB"            
q
bilm/strided_slice_6/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
ъ
bilm/strided_slice_6StridedSlice1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/transpose_1bilm/strided_slice_6/stackbilm/strided_slice_6/stack_1bilm/strided_slice_6/stack_2*
end_mask*5
_output_shapes#
!:                  ђ*

begin_mask*
Index0*
T0
Ъ
bilm/ReverseSequence_2ReverseSequencebilm/strided_slice_6Sum*
T0*

Tlen0*
seq_dim*5
_output_shapes#
!:                  ђ
M
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ќ
concatConcatV2bilm/strided_slice_3bilm/ReverseSequence_1concat/axis*5
_output_shapes#
!:                  ђ*
N*
T0
O
concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B :
џ
concat_1ConcatV2bilm/strided_slice_4bilm/ReverseSequence_2concat_1/axis*5
_output_shapes#
!:                  ђ*
N*
T0
Y
aggregation/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
а
aggregation/concatConcatV2bilm/Reshape_1bilm/Reshape_1aggregation/concat/axis*5
_output_shapes#
!:                  ђ*
N*
T0
џ
%aggregation/weights/Initializer/ConstConst*
valueB*    *&
_class
loc:@aggregation/weights*
dtype0*
_output_shapes
:
д
aggregation/weightsVarHandleOp*
shape:*&
_class
loc:@aggregation/weights*
dtype0*
_output_shapes
: *$
shared_nameaggregation/weights
w
4aggregation/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpaggregation/weights*
_output_shapes
: 
w
aggregation/weights/AssignAssignVariableOpaggregation/weights%aggregation/weights/Initializer/Const*
dtype0
w
'aggregation/weights/Read/ReadVariableOpReadVariableOpaggregation/weights*
dtype0*
_output_shapes
:
њ
%aggregation/scaling/Initializer/ConstConst*
valueB
 *  ђ?*&
_class
loc:@aggregation/scaling*
dtype0*
_output_shapes
: 
б
aggregation/scalingVarHandleOp*
shape: *&
_class
loc:@aggregation/scaling*
dtype0*
_output_shapes
: *$
shared_nameaggregation/scaling
w
4aggregation/scaling/IsInitialized/VarIsInitializedOpVarIsInitializedOpaggregation/scaling*
_output_shapes
: 
w
aggregation/scaling/AssignAssignVariableOpaggregation/scaling%aggregation/scaling/Initializer/Const*
dtype0
s
'aggregation/scaling/Read/ReadVariableOpReadVariableOpaggregation/scaling*
dtype0*
_output_shapes
: 
j
aggregation/ReadVariableOpReadVariableOpaggregation/weights*
dtype0*
_output_shapes
:
_
aggregation/SoftmaxSoftmaxaggregation/ReadVariableOp*
_output_shapes
:*
T0
i
aggregation/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
k
!aggregation/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
k
!aggregation/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
в
aggregation/strided_sliceStridedSliceaggregation/Softmaxaggregation/strided_slice/stack!aggregation/strided_slice/stack_1!aggregation/strided_slice/stack_2*
T0*
_output_shapes
: *
Index0*
shrink_axis_mask
Ё
aggregation/mulMulaggregation/strided_sliceaggregation/concat*
T0*5
_output_shapes#
!:                  ђ
V
aggregation/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
aggregation/addAddV2aggregation/add/xaggregation/mul*
T0*5
_output_shapes#
!:                  ђ
k
!aggregation/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
m
#aggregation/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
m
#aggregation/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
з
aggregation/strided_slice_1StridedSliceaggregation/Softmax!aggregation/strided_slice_1/stack#aggregation/strided_slice_1/stack_1#aggregation/strided_slice_1/stack_2*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: 
}
aggregation/mul_1Mulaggregation/strided_slice_1concat*5
_output_shapes#
!:                  ђ*
T0
~
aggregation/add_1AddV2aggregation/addaggregation/mul_1*5
_output_shapes#
!:                  ђ*
T0
k
!aggregation/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
m
#aggregation/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
m
#aggregation/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
з
aggregation/strided_slice_2StridedSliceaggregation/Softmax!aggregation/strided_slice_2/stack#aggregation/strided_slice_2/stack_1#aggregation/strided_slice_2/stack_2*
Index0*
shrink_axis_mask*
T0*
_output_shapes
: 

aggregation/mul_2Mulaggregation/strided_slice_2concat_1*
T0*5
_output_shapes#
!:                  ђ
ђ
aggregation/add_2AddV2aggregation/add_1aggregation/mul_2*
T0*5
_output_shapes#
!:                  ђ
h
aggregation/ReadVariableOp_1ReadVariableOpaggregation/scaling*
dtype0*
_output_shapes
: 
Ѕ
aggregation/mul_3Mulaggregation/ReadVariableOp_1aggregation/add_2*5
_output_shapes#
!:                  ђ*
T0
\
SequenceMask/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Q
SequenceMask/MaxMaxSumSequenceMask/Const*
_output_shapes
: *
T0
V
SequenceMask/Const_1Const*
value	B : *
dtype0*
_output_shapes
: 
h
SequenceMask/MaximumMaximumSequenceMask/Const_1SequenceMask/Max*
T0*
_output_shapes
: 
V
SequenceMask/Const_2Const*
dtype0*
_output_shapes
: *
value	B : 
V
SequenceMask/Const_3Const*
value	B :*
dtype0*
_output_shapes
: 
ѓ
SequenceMask/RangeRangeSequenceMask/Const_2SequenceMask/MaximumSequenceMask/Const_3*#
_output_shapes
:         
f
SequenceMask/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
y
SequenceMask/ExpandDims
ExpandDimsSumSequenceMask/ExpandDims/dim*
T0*'
_output_shapes
:         
s
SequenceMask/CastCastSequenceMask/ExpandDims*

SrcT0*

DstT0*'
_output_shapes
:         
{
SequenceMask/LessLessSequenceMask/RangeSequenceMask/Cast*
T0*0
_output_shapes
:                  
x
SequenceMask/Cast_1CastSequenceMask/Less*

SrcT0
*

DstT0*0
_output_shapes
:                  
Y
ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
|

ExpandDims
ExpandDimsSequenceMask/Cast_1ExpandDims/dim*
T0*4
_output_shapes"
 :                  
i
mulMulaggregation/mul_3
ExpandDims*
T0*5
_output_shapes#
!:                  ђ
Y
Sum_1/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
]
Sum_1SummulSum_1/reduction_indices*(
_output_shapes
:         ђ*
T0
K
	Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
P
MaximumMaximumSum	Maximum/y*#
_output_shapes
:         *
T0
R
ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
g
ExpandDims_1
ExpandDimsMaximumExpandDims_1/dim*
T0*'
_output_shapes
:         
^
ToFloatCastExpandDims_1*

SrcT0*

DstT0*'
_output_shapes
:         
U
truedivRealDivSum_1ToFloat*(
_output_shapes
:         ђ*
T0"Х"┤ъ
while_contextАъЮъ
Я
map/while/while_context
*map/while/LoopCond:02map/while/Merge:0:map/while/Identity:0Bmap/while/Exit:0Bmap/while/Exit_1:0Bmap/while/Exit_2:0Jэ
map/TensorArray:0
@map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map/TensorArray_1:0
map/strided_slice:0
map/while/DecodeRaw:0
map/while/Enter:0
map/while/Enter_1:0
map/while/Enter_2:0
map/while/Exit:0
map/while/Exit_1:0
map/while/Exit_2:0
map/while/Fill/dims:0
map/while/Fill/value:0
map/while/Fill:0
map/while/Identity:0
map/while/Identity_1:0
map/while/Identity_2:0
map/while/Less/Enter:0
map/while/Less:0
map/while/Less_1:0
map/while/LogicalAnd:0
map/while/LoopCond:0
map/while/Merge:0
map/while/Merge:1
map/while/Merge_1:0
map/while/Merge_1:1
map/while/Merge_2:0
map/while/Merge_2:1
map/while/NextIteration:0
map/while/NextIteration_1:0
map/while/NextIteration_2:0
map/while/Shape:0
map/while/Switch:0
map/while/Switch:1
map/while/Switch_1:0
map/while/Switch_1:1
map/while/Switch_2:0
map/while/Switch_2:1
#map/while/TensorArrayReadV3/Enter:0
%map/while/TensorArrayReadV3/Enter_1:0
map/while/TensorArrayReadV3:0
5map/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
/map/while/TensorArrayWrite/TensorArrayWriteV3:0
map/while/ToInt32:0
map/while/add/y:0
map/while/add:0
map/while/add_1/y:0
map/while/add_1:0
map/while/concat/axis:0
map/while/concat/values_0:0
map/while/concat/values_2:0
map/while/concat:0
map/while/strided_slice/stack:0
!map/while/strided_slice/stack_1:0
!map/while/strided_slice/stack_2:0
map/while/strided_slice:0
!map/while/strided_slice_1/stack:0
#map/while/strided_slice_1/stack_1:0
#map/while/strided_slice_1/stack_2:0
map/while/strided_slice_1:0
map/while/sub/x:0
map/while/sub:0
map/while/sub_1/y:0
map/while/sub_1:0i
@map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0%map/while/TensorArrayReadV3/Enter_1:0-
map/strided_slice:0map/while/Less/Enter:0L
map/TensorArray_1:05map/while/TensorArrayWrite/TensorArrayWriteV3/Enter:08
map/TensorArray:0#map/while/TensorArrayReadV3/Enter:0Rmap/while/Enter:0Rmap/while/Enter_1:0Rmap/while/Enter_2:0Zmap/strided_slice:0
ЉC
9bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/while_context *6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/LoopCond:023bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge:0:6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity:0B2bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Exit:0B4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Exit_1:0B4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Exit_2:0B4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Exit_3:0B4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Exit_4:0Jн<
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/CheckSeqLen:0
/bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Minimum:0
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArray:0
bbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArray_1:0
Dbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Read/Identity:0
Fbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Read/Identity:0
Qbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Read/Identity:0
7bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1:0
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter_1:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter_2:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter_3:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter_4:0
2bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Exit:0
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Exit_1:0
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Exit_2:0
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Exit_3:0
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Exit_4:0
@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqual/Enter:0
:bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqual:0
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity:0
8bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_1:0
8bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_2:0
8bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_3:0
8bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Identity_4:0
8bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Less/Enter:0
2bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Less:0
:bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Less_1/Enter:0
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Less_1:0
8bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/LogicalAnd:0
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/LoopCond:0
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge:0
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge:1
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_1:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_1:1
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_2:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_2:1
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_3:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_3:1
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_4:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Merge_4:1
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_1:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_2:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_3:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_4:0
:bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Select/Enter:0
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Select:0
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Select_1:0
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Select_2:0
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch:0
4bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch:1
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_1:0
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_1:1
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_2:0
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_2:1
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_3:0
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_3:1
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_4:0
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Switch_4:1
Ebilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3/Enter:0
Gbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3/Enter_1:0
?bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3:0
Wbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Qbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/add/y:0
1bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/add:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/add_1/y:0
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/add_1:0
Ebilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/BiasAdd/Enter:0
?bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/BiasAdd:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Const:0
Dbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul/Enter:0
>bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul:0
Fbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul_1/Enter:0
@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul_1:0
?bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Sigmoid:0
Abilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Sigmoid_1:0
Abilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Sigmoid_2:0
<bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Tanh:0
>bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Tanh_1:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/add/y:0
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/add:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/add_1:0
Obilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/Minimum/y:0
Mbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/Minimum:0
Gbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/y:0
Ebilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value:0
Qbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/Minimum/y:0
Obilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/Minimum:0
Ibilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/y:0
Gbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1:0
Cbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/concat/axis:0
>bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/concat:0
;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mul:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mul_1:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mul_2:0
Gbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split/split_dim:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split:1
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split:2
=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split:3
-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/zeros:0s
7bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1:08bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Less/Enter:0Г
bbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0Gbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3/Enter_1:0ј
Fbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Read/Identity:0Dbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul/Enter:0w
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/CheckSeqLen:0@bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqual/Enter:0k
-bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/zeros:0:bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Select/Enter:0m
/bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/Minimum:0:bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Less_1/Enter:0~
5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArray_1:0Ebilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3/Enter:0Ї
Dbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Read/Identity:0Ebilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/BiasAdd/Enter:0Џ
Qbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Read/Identity:0Fbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul_1/Enter:0ј
3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/TensorArray:0Wbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0R3bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter:0R5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter_1:0R5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter_2:0R5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter_3:0R5bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/while/Enter_4:0Z7bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1:0
кC
9bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/while_context *6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/LoopCond:023bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge:0:6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity:0B2bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Exit:0B4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Exit_1:0B4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Exit_2:0B4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Exit_3:0B4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Exit_4:0JЅ=
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/CheckSeqLen:0
/bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Minimum:0
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArray:0
bbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArray_1:0
Dbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Read/Identity:0
Fbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Read/Identity:0
Qbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Read/Identity:0
7bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1:0
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter_1:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter_2:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter_3:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter_4:0
2bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Exit:0
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Exit_1:0
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Exit_2:0
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Exit_3:0
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Exit_4:0
@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqual/Enter:0
:bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqual:0
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity:0
8bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_1:0
8bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_2:0
8bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_3:0
8bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Identity_4:0
8bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Less/Enter:0
2bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Less:0
:bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Less_1/Enter:0
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Less_1:0
8bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/LogicalAnd:0
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/LoopCond:0
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge:0
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge:1
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_1:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_1:1
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_2:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_2:1
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_3:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_3:1
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_4:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Merge_4:1
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_1:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_2:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_3:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_4:0
:bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Select/Enter:0
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Select:0
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Select_1:0
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Select_2:0
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch:0
4bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch:1
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_1:0
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_1:1
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_2:0
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_2:1
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_3:0
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_3:1
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_4:0
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Switch_4:1
Ebilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3/Enter:0
Gbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3/Enter_1:0
?bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3:0
Wbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Qbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add/y:0
1bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add:0
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add_1:0
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add_2/y:0
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/add_2:0
Ebilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/BiasAdd/Enter:0
?bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/BiasAdd:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Const:0
Dbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul/Enter:0
>bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul:0
Fbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul_1/Enter:0
@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul_1:0
?bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Sigmoid:0
Abilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Sigmoid_1:0
Abilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Sigmoid_2:0
<bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Tanh:0
>bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Tanh_1:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/add/y:0
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/add:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/add_1:0
Obilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/Minimum/y:0
Mbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/Minimum:0
Gbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/y:0
Ebilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value:0
Qbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/Minimum/y:0
Obilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/Minimum:0
Ibilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/y:0
Gbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1:0
Cbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/concat/axis:0
>bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/concat:0
;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mul:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mul_1:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mul_2:0
Gbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split/split_dim:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split:0
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split:1
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split:2
=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split:3
-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/zeros:0Џ
Qbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Read/Identity:0Fbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul_1/Enter:0w
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/CheckSeqLen:0@bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqual/Enter:0m
/bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/Minimum:0:bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Less_1/Enter:0s
7bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1:08bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Less/Enter:0ј
3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArray:0Wbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0Г
bbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0Gbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3/Enter_1:0k
-bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/zeros:0:bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Select/Enter:0ј
Fbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Read/Identity:0Dbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul/Enter:0~
5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/TensorArray_1:0Ebilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3/Enter:0Ї
Dbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Read/Identity:0Ebilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/BiasAdd/Enter:0R3bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter:0R5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter_1:0R5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter_2:0R5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter_3:0R5bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/while/Enter_4:0Z7bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1:0
ЉC
9bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/while_context *6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/LoopCond:023bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge:0:6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity:0B2bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Exit:0B4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Exit_1:0B4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Exit_2:0B4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Exit_3:0B4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Exit_4:0Jн<
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/CheckSeqLen:0
/bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Minimum:0
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArray:0
bbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArray_1:0
Dbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Read/Identity:0
Fbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Read/Identity:0
Qbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Read/Identity:0
7bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1:0
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter_1:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter_2:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter_3:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter_4:0
2bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Exit:0
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Exit_1:0
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Exit_2:0
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Exit_3:0
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Exit_4:0
@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqual/Enter:0
:bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqual:0
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity:0
8bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_1:0
8bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_2:0
8bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_3:0
8bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Identity_4:0
8bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Less/Enter:0
2bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Less:0
:bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Less_1/Enter:0
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Less_1:0
8bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/LogicalAnd:0
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/LoopCond:0
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge:0
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge:1
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_1:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_1:1
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_2:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_2:1
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_3:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_3:1
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_4:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Merge_4:1
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_1:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_2:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_3:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/NextIteration_4:0
:bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Select/Enter:0
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Select:0
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Select_1:0
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Select_2:0
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch:0
4bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch:1
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_1:0
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_1:1
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_2:0
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_2:1
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_3:0
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_3:1
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_4:0
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Switch_4:1
Ebilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3/Enter:0
Gbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3/Enter_1:0
?bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3:0
Wbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Qbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/add/y:0
1bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/add:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/add_1/y:0
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/add_1:0
Ebilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/BiasAdd/Enter:0
?bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/BiasAdd:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Const:0
Dbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul/Enter:0
>bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul:0
Fbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul_1/Enter:0
@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul_1:0
?bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Sigmoid:0
Abilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Sigmoid_1:0
Abilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Sigmoid_2:0
<bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Tanh:0
>bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/Tanh_1:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/add/y:0
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/add:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/add_1:0
Obilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/Minimum/y:0
Mbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/Minimum:0
Gbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value/y:0
Ebilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value:0
Qbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/Minimum/y:0
Obilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/Minimum:0
Ibilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1/y:0
Gbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/clip_by_value_1:0
Cbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/concat/axis:0
>bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/concat:0
;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mul:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mul_1:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/mul_2:0
Gbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split/split_dim:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split:1
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split:2
=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/split:3
-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/zeros:0s
7bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1:08bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Less/Enter:0Ї
Dbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Read/Identity:0Ebilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/BiasAdd/Enter:0Џ
Qbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Read/Identity:0Fbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul_1/Enter:0w
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/CheckSeqLen:0@bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/GreaterEqual/Enter:0ј
Fbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Read/Identity:0Dbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/lstm_cell/MatMul/Enter:0~
5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArray_1:0Ebilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3/Enter:0k
-bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/zeros:0:bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Select/Enter:0Г
bbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0Gbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayReadV3/Enter_1:0m
/bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/Minimum:0:bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Less_1/Enter:0ј
3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/TensorArray:0Wbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0R3bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter:0R5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter_1:0R5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter_2:0R5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter_3:0R5bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/while/Enter_4:0Z7bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/strided_slice_1:0
кC
9bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/while_context *6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/LoopCond:023bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge:0:6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity:0B2bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Exit:0B4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Exit_1:0B4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Exit_2:0B4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Exit_3:0B4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Exit_4:0JЅ=
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/CheckSeqLen:0
/bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Minimum:0
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArray:0
bbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArray_1:0
Dbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Read/Identity:0
Fbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Read/Identity:0
Qbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Read/Identity:0
7bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1:0
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter_1:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter_2:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter_3:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter_4:0
2bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Exit:0
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Exit_1:0
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Exit_2:0
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Exit_3:0
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Exit_4:0
@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqual/Enter:0
:bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqual:0
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity:0
8bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_1:0
8bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_2:0
8bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_3:0
8bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Identity_4:0
8bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Less/Enter:0
2bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Less:0
:bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Less_1/Enter:0
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Less_1:0
8bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/LogicalAnd:0
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/LoopCond:0
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge:0
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge:1
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_1:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_1:1
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_2:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_2:1
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_3:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_3:1
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_4:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Merge_4:1
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_1:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_2:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_3:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/NextIteration_4:0
:bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Select/Enter:0
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Select:0
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Select_1:0
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Select_2:0
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch:0
4bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch:1
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_1:0
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_1:1
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_2:0
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_2:1
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_3:0
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_3:1
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_4:0
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Switch_4:1
Ebilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3/Enter:0
Gbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3/Enter_1:0
?bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3:0
Wbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Qbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add/y:0
1bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add:0
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add_1:0
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add_2/y:0
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/add_2:0
Ebilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/BiasAdd/Enter:0
?bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/BiasAdd:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Const:0
Dbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul/Enter:0
>bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul:0
Fbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul_1/Enter:0
@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul_1:0
?bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Sigmoid:0
Abilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Sigmoid_1:0
Abilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Sigmoid_2:0
<bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Tanh:0
>bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/Tanh_1:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/add/y:0
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/add:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/add_1:0
Obilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/Minimum/y:0
Mbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/Minimum:0
Gbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value/y:0
Ebilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value:0
Qbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/Minimum/y:0
Obilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/Minimum:0
Ibilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1/y:0
Gbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/clip_by_value_1:0
Cbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/concat/axis:0
>bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/concat:0
;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mul:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mul_1:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/mul_2:0
Gbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split/split_dim:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split:0
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split:1
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split:2
=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/split:3
-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/zeros:0m
/bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/Minimum:0:bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Less_1/Enter:0Ї
Dbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Read/Identity:0Ebilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/BiasAdd/Enter:0k
-bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/zeros:0:bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Select/Enter:0ј
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArray:0Wbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0~
5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArray_1:0Ebilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3/Enter:0ј
Fbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Read/Identity:0Dbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul/Enter:0Џ
Qbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Read/Identity:0Fbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/lstm_cell/MatMul_1/Enter:0Г
bbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0Gbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/TensorArrayReadV3/Enter_1:0w
3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/CheckSeqLen:0@bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/GreaterEqual/Enter:0s
7bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1:08bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Less/Enter:0R3bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter:0R5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter_1:0R5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter_2:0R5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter_3:0R5bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/while/Enter_4:0Z7bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/strided_slice_1:0"и
trainable_variablesЪю
І
aggregation/weights:0aggregation/weights/Assign)aggregation/weights/Read/ReadVariableOp:0(2'aggregation/weights/Initializer/Const:08
І
aggregation/scaling:0aggregation/scaling/Assign)aggregation/scaling/Read/ReadVariableOp:0(2'aggregation/scaling/Initializer/Const:08"─;
	variablesХ;│;
ё
bilm/char_embed:0bilm/char_embed/Assign%bilm/char_embed/Read/ReadVariableOp:0(2,bilm/char_embed/Initializer/random_uniform:08
ѕ
bilm/CNN/W_cnn_0:0bilm/CNN/W_cnn_0/Assign&bilm/CNN/W_cnn_0/Read/ReadVariableOp:0(2-bilm/CNN/W_cnn_0/Initializer/random_uniform:08
ѕ
bilm/CNN/b_cnn_0:0bilm/CNN/b_cnn_0/Assign&bilm/CNN/b_cnn_0/Read/ReadVariableOp:0(2-bilm/CNN/b_cnn_0/Initializer/random_uniform:08
ѕ
bilm/CNN/W_cnn_1:0bilm/CNN/W_cnn_1/Assign&bilm/CNN/W_cnn_1/Read/ReadVariableOp:0(2-bilm/CNN/W_cnn_1/Initializer/random_uniform:08
ѕ
bilm/CNN/b_cnn_1:0bilm/CNN/b_cnn_1/Assign&bilm/CNN/b_cnn_1/Read/ReadVariableOp:0(2-bilm/CNN/b_cnn_1/Initializer/random_uniform:08
ѕ
bilm/CNN/W_cnn_2:0bilm/CNN/W_cnn_2/Assign&bilm/CNN/W_cnn_2/Read/ReadVariableOp:0(2-bilm/CNN/W_cnn_2/Initializer/random_uniform:08
ѕ
bilm/CNN/b_cnn_2:0bilm/CNN/b_cnn_2/Assign&bilm/CNN/b_cnn_2/Read/ReadVariableOp:0(2-bilm/CNN/b_cnn_2/Initializer/random_uniform:08
ѕ
bilm/CNN/W_cnn_3:0bilm/CNN/W_cnn_3/Assign&bilm/CNN/W_cnn_3/Read/ReadVariableOp:0(2-bilm/CNN/W_cnn_3/Initializer/random_uniform:08
ѕ
bilm/CNN/b_cnn_3:0bilm/CNN/b_cnn_3/Assign&bilm/CNN/b_cnn_3/Read/ReadVariableOp:0(2-bilm/CNN/b_cnn_3/Initializer/random_uniform:08
ѕ
bilm/CNN/W_cnn_4:0bilm/CNN/W_cnn_4/Assign&bilm/CNN/W_cnn_4/Read/ReadVariableOp:0(2-bilm/CNN/W_cnn_4/Initializer/random_uniform:08
ѕ
bilm/CNN/b_cnn_4:0bilm/CNN/b_cnn_4/Assign&bilm/CNN/b_cnn_4/Read/ReadVariableOp:0(2-bilm/CNN/b_cnn_4/Initializer/random_uniform:08
ѕ
bilm/CNN/W_cnn_5:0bilm/CNN/W_cnn_5/Assign&bilm/CNN/W_cnn_5/Read/ReadVariableOp:0(2-bilm/CNN/W_cnn_5/Initializer/random_uniform:08
ѕ
bilm/CNN/b_cnn_5:0bilm/CNN/b_cnn_5/Assign&bilm/CNN/b_cnn_5/Read/ReadVariableOp:0(2-bilm/CNN/b_cnn_5/Initializer/random_uniform:08
ѕ
bilm/CNN/W_cnn_6:0bilm/CNN/W_cnn_6/Assign&bilm/CNN/W_cnn_6/Read/ReadVariableOp:0(2-bilm/CNN/W_cnn_6/Initializer/random_uniform:08
ѕ
bilm/CNN/b_cnn_6:0bilm/CNN/b_cnn_6/Assign&bilm/CNN/b_cnn_6/Read/ReadVariableOp:0(2-bilm/CNN/b_cnn_6/Initializer/random_uniform:08
ц
bilm/CNN_high_0/W_carry:0bilm/CNN_high_0/W_carry/Assign-bilm/CNN_high_0/W_carry/Read/ReadVariableOp:0(24bilm/CNN_high_0/W_carry/Initializer/random_uniform:08
ц
bilm/CNN_high_0/b_carry:0bilm/CNN_high_0/b_carry/Assign-bilm/CNN_high_0/b_carry/Read/ReadVariableOp:0(24bilm/CNN_high_0/b_carry/Initializer/random_uniform:08
┤
bilm/CNN_high_0/W_transform:0"bilm/CNN_high_0/W_transform/Assign1bilm/CNN_high_0/W_transform/Read/ReadVariableOp:0(28bilm/CNN_high_0/W_transform/Initializer/random_uniform:08
┤
bilm/CNN_high_0/b_transform:0"bilm/CNN_high_0/b_transform/Assign1bilm/CNN_high_0/b_transform/Read/ReadVariableOp:0(28bilm/CNN_high_0/b_transform/Initializer/random_uniform:08
ц
bilm/CNN_high_1/W_carry:0bilm/CNN_high_1/W_carry/Assign-bilm/CNN_high_1/W_carry/Read/ReadVariableOp:0(24bilm/CNN_high_1/W_carry/Initializer/random_uniform:08
ц
bilm/CNN_high_1/b_carry:0bilm/CNN_high_1/b_carry/Assign-bilm/CNN_high_1/b_carry/Read/ReadVariableOp:0(24bilm/CNN_high_1/b_carry/Initializer/random_uniform:08
┤
bilm/CNN_high_1/W_transform:0"bilm/CNN_high_1/W_transform/Assign1bilm/CNN_high_1/W_transform/Read/ReadVariableOp:0(28bilm/CNN_high_1/W_transform/Initializer/random_uniform:08
┤
bilm/CNN_high_1/b_transform:0"bilm/CNN_high_1/b_transform/Assign1bilm/CNN_high_1/b_transform/Read/ReadVariableOp:0(28bilm/CNN_high_1/b_transform/Initializer/random_uniform:08
ў
bilm/CNN_proj/W_proj:0bilm/CNN_proj/W_proj/Assign*bilm/CNN_proj/W_proj/Read/ReadVariableOp:0(21bilm/CNN_proj/W_proj/Initializer/random_uniform:08
ў
bilm/CNN_proj/b_proj:0bilm/CNN_proj/b_proj/Assign*bilm/CNN_proj/b_proj/Read/ReadVariableOp:0(21bilm/CNN_proj/b_proj/Initializer/random_uniform:08
џ
8bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel:0=bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/AssignFbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Read/Identity:0(2Sbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform:08
Ѕ
6bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias:0;bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/AssignDbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Read/Identity:0(2Hbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Initializer/zeros:08
к
Cbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel:0Hbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/AssignQbilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Read/Identity:0(2^bilm/RNN_0/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform:08
џ
8bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel:0=bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/AssignFbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Read/Identity:0(2Sbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform:08
Ѕ
6bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias:0;bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/AssignDbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Read/Identity:0(2Hbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Initializer/zeros:08
к
Cbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel:0Hbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/AssignQbilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Read/Identity:0(2^bilm/RNN_0/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform:08
џ
8bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel:0=bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/AssignFbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Read/Identity:0(2Sbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/kernel/Initializer/random_uniform:08
Ѕ
6bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias:0;bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/AssignDbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Read/Identity:0(2Hbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/bias/Initializer/zeros:08
к
Cbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel:0Hbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/AssignQbilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Read/Identity:0(2^bilm/RNN_1/RNN/MultiRNNCell/Cell0/rnn/lstm_cell/projection/kernel/Initializer/random_uniform:08
џ
8bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel:0=bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/AssignFbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Read/Identity:0(2Sbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/kernel/Initializer/random_uniform:08
Ѕ
6bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias:0;bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/AssignDbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Read/Identity:0(2Hbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/bias/Initializer/zeros:08
к
Cbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel:0Hbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/AssignQbilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Read/Identity:0(2^bilm/RNN_1/RNN/MultiRNNCell/Cell1/rnn/lstm_cell/projection/kernel/Initializer/random_uniform:08
І
aggregation/weights:0aggregation/weights/Assign)aggregation/weights/Read/ReadVariableOp:0(2'aggregation/weights/Initializer/Const:08
І
aggregation/scaling:0aggregation/scaling/Assign)aggregation/scaling/Read/ReadVariableOp:0(2'aggregation/scaling/Initializer/Const:08*┘
word_emb╠
A
word_emb5
bilm/Reshape_1:0                  ђ
(
sequence_len
Sum:0         ,
default!
	truediv:0         ђ>
lstm_outputs1-
concat:0                  ђA
word_emb5
bilm/Reshape_1:0                  ђ@
lstm_outputs2/

concat_1:0                  ђ(
sequence_len
Sum:0         @
elmo8
aggregation/mul_3:0                  ђ*¤
tokens─
9
tokens/
SparseToDense:0                  
(
sequence_len
Sum:0         A
word_emb5
bilm/Reshape_1:0                  ђ@
lstm_outputs2/

concat_1:0                  ђ(
sequence_len
Sum:0         @
elmo8
aggregation/mul_3:0                  ђ,
default!
	truediv:0         ђ>
lstm_outputs1-
concat:0                  ђ*ј
defaultѓ
!
text
text:0         A
word_emb5
bilm/Reshape_1:0                  ђ@
lstm_outputs2/

concat_1:0                  ђ(
sequence_len
Sum:0         @
elmo8
aggregation/mul_3:0                  ђ,
default!
	truediv:0         ђ>
lstm_outputs1-
concat:0                  ђ