//============================================================================
// Dataflow type definitions for the systolic array
//============================================================================

`timescale 1ns / 1ps

package tensor_pkg;

    typedef enum logic [1:0] {
        DF_WEIGHT_STATIONARY = 2'b00,
        DF_OUTPUT_STATIONARY = 2'b01,
        DF_INPUT_STATIONARY  = 2'b10,
        DF_TENSOR_CORE       = 2'b11
    } dataflow_t;

    // 4-bit Tensor ISA opcodes
    localparam logic [3:0] OP_NOP   = 4'b0000;
    localparam logic [3:0] OP_LOAD  = 4'b0001;
    localparam logic [3:0] OP_STORE = 4'b0010;
    localparam logic [3:0] OP_ADD   = 4'b0011;
    localparam logic [3:0] OP_MUL   = 4'b0100;
    localparam logic [3:0] OP_MMA   = 4'b0101;
    localparam logic [3:0] OP_MAX   = 4'b0110;
    localparam logic [3:0] OP_EXP   = 4'b0111;
    localparam logic [3:0] OP_SCALE = 4'b1000;
    localparam logic [3:0] OP_REDUCE= 4'b1001;
    localparam logic [3:0] OP_SHFL  = 4'b1010;
    localparam logic [3:0] OP_MOV   = 4'b1011;
    localparam logic [3:0] OP_CMP   = 4'b1100;
    localparam logic [3:0] OP_BR    = 4'b1101;
    localparam logic [3:0] OP_SYNC  = 4'b1110;
    localparam logic [3:0] OP_HALT  = 4'b1111;

endpackage
