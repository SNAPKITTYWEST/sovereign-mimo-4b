//============================================================================
// Tensor Core — FP16 inputs, FP32 accumulate
// C = A × B + C (matrix multiply-accumulate)
// M×N×K parameterized (default 16×8×16)
//============================================================================

`timescale 1ns / 1ps

module tensor_core_fp16 #(
    parameter int M = 16,
    parameter int N = 8,
    parameter int K = 16
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        done,
    output logic        busy,

    input  logic [15:0] a_matrix [M][K],
    input  logic [15:0] b_matrix [K][N],
    input  logic [31:0] c_in     [M][N],
    output logic [31:0] c_out    [M][N]
);

    logic [31:0] acc [M][N];
    logic [$clog2(K+1)-1:0] k_cnt;
    logic running;

    function automatic logic [31:0] fp16_mul_fp32(input logic [15:0] a, input logic [15:0] b);
        return $signed(a) * $signed(b);
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running <= 0;
            done    <= 0;
            k_cnt   <= 0;
        end else begin
            done <= 0;
            if (start && !running) begin
                running <= 1;
                k_cnt   <= 0;
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++)
                        acc[i][j] <= c_in[i][j];
            end else if (running) begin
                if (k_cnt == K-1) begin
                    running <= 0;
                    done    <= 1;
                end else
                    k_cnt <= k_cnt + 1;
            end
        end
    end

    assign busy = running;

    genvar gi, gj;
    generate
        for (gi = 0; gi < M; gi++) begin : gen_row
            for (gj = 0; gj < N; gj++) begin : gen_col
                always_ff @(posedge clk) begin
                    if (running)
                        acc[gi][gj] <= acc[gi][gj] +
                            fp16_mul_fp32(a_matrix[gi][k_cnt], b_matrix[k_cnt][gj]);
                end
            end
        end
    endgenerate

    always_comb begin
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                c_out[i][j] = acc[i][j];
    end

endmodule
