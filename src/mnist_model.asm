; mnist_model.asm
section .data
    ; File path formats and strings
    train_path_fmt: db "image/train/%d/%d.jpg", 0
    test_path_fmt:  db "image/test/%d/%d.jpg", 0
    read_mode: db "rb", 0
    pixel_fmt: db "%hhu", 0
    
    ; Print formats
    epoch_fmt: db "Epoch %d/%d", 10, 0
    train_prog_fmt: db "Training progress: %d/%d - Loss: %f", 10, 0
    accuracy_fmt: db "Accuracy: %.2f%%", 10, 0
    pred_fmt: db "Predicted digit: %d", 10, 0
    error_fmt: db "Error loading image", 10, 0

    ; Constants
    INPUT_SIZE: equ 784    ; 28x28
    HIDDEN_SIZE: equ 128
    OUTPUT_SIZE: equ 10
    learning_rate: dq 0.01
    NUM_EPOCHS: equ 10
    zero: dq 0.0
    one: dq 1.0
    
    ; Memory address computation constants
    hidden_layer_size: equ HIDDEN_SIZE * 8
    input_hidden_size: equ INPUT_SIZE * HIDDEN_SIZE * 8
    hidden_output_size: equ HIDDEN_SIZE * OUTPUT_SIZE * 8

section .bss
    align 32
    ; Network parameters
    input_hidden: resq INPUT_SIZE * HIDDEN_SIZE
    hidden_bias: resq HIDDEN_SIZE
    hidden_output: resq HIDDEN_SIZE * OUTPUT_SIZE
    output_bias: resq OUTPUT_SIZE
    
    ; Activation storage
    input_layer: resq INPUT_SIZE
    hidden_layer: resq HIDDEN_SIZE
    output_layer: resq OUTPUT_SIZE
    
    ; Gradient storage
    hidden_gradients: resq HIDDEN_SIZE
    output_gradients: resq OUTPUT_SIZE
    
    ; Temporary storage
    temp_storage: resq 1024

section .text
    global main
    extern printf
    extern scanf
    extern malloc
    extern free
    extern fopen
    extern fclose
    extern fscanf
    extern exp
    extern log
    
main:
    push rbp
    mov rbp, rsp
    sub rsp, 64             ; Reserve stack space
    
    ; Initialize random number generator seed
    rdtsc
    mov [rsp], rax
    
    ; Initialize network weights
    call initialize_weights
    
    ; Training loop
    xor r12, r12           ; epoch counter
.epoch_loop:
    ; Print epoch progress
    mov rdi, epoch_fmt
    mov rsi, r12
    mov rdx, NUM_EPOCHS
    call printf
    
    ; Train on all images in epoch
    xor r13, r13           ; image counter
.train_loop:
    ; Load training image
    mov rdi, train_path_fmt
    mov rsi, r13
    mov rdx, r13
    call load_image
    test rax, rax
    jz .train_error
    
    ; Forward pass
    call forward_propagation
    
    ; Backward pass
    call backward_propagation
    
    ; Update weights
    call update_weights
    
    ; Print training progress
    mov rdi, train_prog_fmt
    mov rsi, r13
    mov rdx, 10000         ; total images
    movsd xmm0, [rsp+8]    ; current loss
    call printf
    
    inc r13
    cmp r13, 10000
    jl .train_loop
    
    inc r12
    cmp r12, NUM_EPOCHS
    jl .epoch_loop
    
    ; Handle testing if image provided
    cmp qword [rbp+16], 2  ; check argc
    jl .exit
    
    ; Load and process test image
    mov rdi, [rbp+24]      ; argv[1]
    call load_image
    test rax, rax
    jz .exit
    
    call forward_propagation
    call get_prediction
    
    ; Print prediction
    mov rdi, pred_fmt
    mov rsi, rax
    call printf
    
.exit:
    leave
    ret
    
.train_error:
    mov rdi, error_fmt
    call printf
    jmp .exit

initialize_weights:
    push rbp
    mov rbp, rsp
    push rbx
    
    ; Initialize input->hidden weights
    mov rbx, input_hidden
    mov rcx, INPUT_SIZE * HIDDEN_SIZE
.init_ih_loop:
    call generate_random_weight
    movsd [rbx], xmm0
    add rbx, 8
    loop .init_ih_loop
    
    ; Initialize hidden biases
    mov rbx, hidden_bias
    mov rcx, HIDDEN_SIZE
.init_hb_loop:
    call generate_random_weight
    movsd [rbx], xmm0
    add rbx, 8
    loop .init_hb_loop
    
    ; Initialize hidden->output weights
    mov rbx, hidden_output
    mov rcx, HIDDEN_SIZE * OUTPUT_SIZE
.init_ho_loop:
    call generate_random_weight
    movsd [rbx], xmm0
    add rbx, 8
    loop .init_ho_loop
    
    ; Initialize output biases
    mov rbx, output_bias
    mov rcx, OUTPUT_SIZE
.init_ob_loop:
    call generate_random_weight
    movsd [rbx], xmm0
    add rbx, 8
    loop .init_ob_loop
    
    pop rbx
    leave
    ret

generate_random_weight:
    push rbp
    mov rbp, rsp
    
    ; Generate random number using RDRAND
    rdrand rax
    
    ; Convert to float between -0.5 and 0.5
    cvtsi2sd xmm0, rax
    divsd xmm0, [rel one]
    mulsd xmm0, [rel learning_rate]
    
    leave
    ret

load_image:
    push rbp
    mov rbp, rsp
    sub rsp, 32
    
    ; Open file
    mov rsi, read_mode
    call fopen
    test rax, rax
    jz .error
    
    mov [rsp], rax        ; Save file handle
    
    ; Read pixel data
    mov rbx, input_layer
    mov r12, INPUT_SIZE
.read_loop:
    mov rdi, [rsp]
    mov rsi, pixel_fmt
    mov rdx, temp_storage
    push rbx
    push r12
    call fscanf
    pop r12
    pop rbx
    
    ; Convert to float and normalize (0-1)
    movzx eax, byte [temp_storage]
    cvtsi2sd xmm0, eax
    divsd xmm0, [rel one]
    movsd [rbx], xmm0
    
    add rbx, 8
    dec r12
    jnz .read_loop
    
    ; Close file
    mov rdi, [rsp]
    call fclose
    
    mov rax, 1          ; Success
    leave
    ret
    
.error:
    xor rax, rax        ; Failure
    leave
    ret

forward_propagation:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    ; Input -> Hidden layer
    mov r12, HIDDEN_SIZE    ; Hidden neuron counter
.hidden_loop:
    pxor xmm0, xmm0        ; Accumulator
    mov r13, INPUT_SIZE     ; Input neuron counter
    mov r14, input_layer
    mov r15, input_hidden
    add r15, r12
.input_loop:
    movsd xmm1, [r14]
    mulsd xmm1, [r15]
    addsd xmm0, xmm1
    add r14, 8
    add r15, HIDDEN_SIZE * 8
    dec r13
    jnz .input_loop
    
    ; Add bias and apply ReLU
    addsd xmm0, [hidden_bias + r12 * 8]
    maxsd xmm0, [rel zero]
    movsd [hidden_layer + r12 * 8], xmm0
    
    dec r12
    jnz .hidden_loop
    
    ; Hidden -> Output layer
    mov r12, OUTPUT_SIZE    ; Output neuron counter
.output_loop:
    pxor xmm0, xmm0        ; Accumulator
    mov r13, HIDDEN_SIZE    ; Hidden neuron counter
    mov r14, hidden_layer
    mov r15, hidden_output
    add r15, r12
.hidden_sum_loop:
    movsd xmm1, [r14]
    mulsd xmm1, [r15]
    addsd xmm0, xmm1
    add r14, 8
    add r15, OUTPUT_SIZE * 8
    dec r13
    jnz .hidden_sum_loop
    
    ; Add bias and store
    addsd xmm0, [output_bias + r12 * 8]
    movsd [output_layer + r12 * 8], xmm0
    
    dec r12
    jnz .output_loop
    
    ; Apply softmax to output layer
    call apply_softmax
    
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    leave
    ret

apply_softmax:
    push rbp
    mov rbp, rsp
    
    ; Find maximum value for numerical stability
    movsd xmm0, [output_layer]    ; max value
    mov rcx, 1
.max_loop:
    movsd xmm1, [output_layer + rcx * 8]
    maxsd xmm0, xmm1
    inc rcx
    cmp rcx, OUTPUT_SIZE
    jl .max_loop
    
    ; Calculate exp(x - max) and sum
    pxor xmm2, xmm2                ; sum
    mov rcx, OUTPUT_SIZE
.exp_loop:
    movsd xmm1, [output_layer + rcx * 8 - 8]
    subsd xmm1, xmm0
    
    ; Calculate exp(x)
    sub rsp, 32
    movsd [rsp], xmm1
    movsd [rsp+8], xmm2
    call exp
    movsd xmm2, [rsp+8]
    add rsp, 32
    
    addsd xmm2, xmm0               ; Add to sum
    movsd [output_layer + rcx * 8 - 8], xmm0  ; Store exp value
    loop .exp_loop
    
    ; Normalize by sum
    mov rcx, OUTPUT_SIZE
.normalize_loop:
    movsd xmm0, [output_layer + rcx * 8 - 8]
    divsd xmm0, xmm2
    movsd [output_layer + rcx * 8 - 8], xmm0
    loop .normalize_loop
    
    leave
    ret

backward_propagation:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    ; Calculate output layer gradients
    mov rcx, OUTPUT_SIZE
.output_grad_loop:
    movsd xmm0, [output_layer + rcx * 8 - 8]    ; predicted
    subsd xmm0, xmm1                            ; target (1 for correct class, 0 for others)
    movsd [output_gradients + rcx * 8 - 8], xmm0
    loop .output_grad_loop
    
    ; Calculate hidden layer gradients
    mov r12, HIDDEN_SIZE    ; Hidden neuron counter
.hidden_grad_loop:
    pxor xmm0, xmm0        ; Accumulator
    mov r13, OUTPUT_SIZE    ; Output neuron counter
    mov r14, output_gradients
    mov r15, hidden_output
    add r15, r12
.grad_sum_loop:
    movsd xmm1, [r14]
    mulsd xmm1, [r15]
    addsd xmm0, xmm1
    add r14, 8
    add r15, HIDDEN_SIZE * 8
    dec r13
    jnz .grad_sum_loop
    
    ; Multiply by ReLU derivative
    movsd xmm1, [hidden_layer + r12 * 8]
    xorpd xmm2, xmm2
    cmpsd xmm2, xmm1, 1    ; Compare if > 0
    andpd xmm0, xmm2       ; Zero gradient if input <= 0
    
    movsd [hidden_gradients + r12 * 8], xmm0
    
    dec r12
    jnz .hidden_grad_loop
    
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    leave
    ret

update_weights:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    ; Update hidden->output weights
    mov r12, HIDDEN_SIZE
.update_ho_loop:
    mov r13, OUTPUT_SIZE
.update_ho_inner:
    mov rax, r12
    mul r13
    mov rbx, rax        ; offset = r12 * OUTPUT_SIZE + r13
    
    movsd xmm0, [hidden_layer + r12 * 8]
    mulsd xmm0, [output_gradients + r13 * 8]
    mulsd xmm0, [learning_rate]
    movsd xmm1, [hidden_output + rbx * 8]
    subsd xmm1, xmm0
    movsd [hidden_output + rbx * 8], xmm1
    
    dec r13
    jnz .update_ho_inner
    dec r12
    jnz .update_ho_loop
    
    ; Update output biases
    mov rcx, OUTPUT_SIZE
.update_ob_loop:
    movsd xmm0, [output_gradients + rcx * 8 - 8]
    mulsd xmm0, [learning_rate]
    movsd xmm1, [output_bias + rcx * 8 - 8]
    subsd xmm1, xmm0
    movsd [output_bias + rcx * 8 - 8], xmm1
    loop .update_ob_loop
    
    ; Update input->hidden weights
    mov r12, INPUT_SIZE
.update_ih_loop:
    mov r13, HIDDEN_SIZE
.update_ih_inner:
    mov rax, r12
    mul r13
    mov rbx, rax        ; offset = r12 * HIDDEN_SIZE + r13
    
    movsd xmm0, [input_layer + r12 * 8]
    mulsd xmm0, [hidden_gradients + r13 * 8]
    mulsd xmm0, [learning_rate]
    movsd xmm1, [input_hidden + rbx * 8]
    subsd xmm1, xmm0
    movsd [input_hidden + rbx * 8], xmm1
    
    dec r13
    jnz .update_ih_inner
    dec r12
    jnz .update_ih_loop
    
    ; Update hidden biases
    mov rcx, HIDDEN_SIZE
.update_hb_loop:
    movsd xmm0, [hidden_gradients + rcx * 8 - 8]
    mulsd xmm0, [learning_rate]
    movsd xmm1, [hidden_bias + rcx * 8 - 8]
    subsd xmm1, xmm0
    movsd [hidden_bias + rcx * 8 - 8], xmm1
    loop .update_hb_loop
    
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    leave
    ret

get_prediction:
    push rbp
    mov rbp, rsp
    
    ; Find index of maximum value in output_layer
    xor rax, rax            ; max index
    movsd xmm0, [output_layer]  ; max value
    mov rcx, 1              ; current index
    
.max_loop:
    movsd xmm1, [output_layer + rcx * 8]
    ucomisd xmm1, xmm0
    jbe .next_comp
    
    ; Update max
    movsd xmm0, xmm1
    mov rax, rcx
    
.next_comp:
    inc rcx
    cmp rcx, OUTPUT_SIZE
    jl .max_loop
    
    leave
    ret

section .data
    align 8
    ; Additional constants
    float_format: db "%.6f", 10, 0
    debug_msg: db "Debug value: %.6f", 10, 0
    malloc_error_msg: db "Memory allocation failed", 10, 0
    file_error_msg: db "File operation failed", 10, 0

section .bss
    align 32
    ; Additional temporary storage
    debug_buffer: resq 1
    temp_vector: resq 128
    batch_gradients: resq 1024

section .text
print_debug:
    ; Utility function to print debug values
    push rbp
    mov rbp, rsp
    
    mov rdi, debug_msg
    movsd xmm0, [debug_buffer]
    mov rax, 1          ; one floating point argument
    call printf
    
    leave
    ret

handle_error:
    ; General error handling routine
    push rbp
    mov rbp, rsp
    
    mov rdi, [rbp+16]   ; error message pointer
    call printf
    
    mov rax, 1          ; error exit code
    leave
    ret

calculate_loss:
    ; Calculate cross-entropy loss
    push rbp
    mov rbp, rsp
    
    ; Initialize loss to 0
    pxor xmm0, xmm0
    movsd [debug_buffer], xmm0
    
    ; Sum -y_true * log(y_pred)
    mov rcx, OUTPUT_SIZE
.loss_loop:
    movsd xmm0, [output_layer + rcx * 8 - 8]
    
    ; Calculate log(y_pred)
    sub rsp, 32
    movsd [rsp], xmm0
    call log
    add rsp, 32
    
    ; Multiply by -y_true (1 for correct class, 0 for others)
    mulsd xmm0, [temp_vector + rcx * 8 - 8]
    xorpd xmm1, xmm1
    subsd xmm1, xmm0
    movsd xmm0, xmm1
    
    ; Add to total loss
    movsd xmm1, [debug_buffer]
    addsd xmm1, xmm0
    movsd [debug_buffer], xmm1
    
    loop .loss_loop
    
    ; Return loss in xmm0
    movsd xmm0, [debug_buffer]
    
    leave
    ret

section .data
    align 8
    ; Configuration constants
    BATCH_SIZE: equ 32
    MIN_GRAD: dq 0.0001    ; Minimum gradient for numerical stability
    MAX_GRAD: dq 1.0       ; Maximum gradient to prevent explosion

section .text
clip_gradients:
    ; Utility function to prevent gradient explosion
    push rbp
    mov rbp, rsp
    
    mov rcx, HIDDEN_SIZE
.clip_loop:
    movsd xmm0, [hidden_gradients + rcx * 8 - 8]
    
    ; Clip minimum
    maxsd xmm0, [MIN_GRAD]
    
    ; Clip maximum
    minsd xmm0, [MAX_GRAD]
    
    movsd [hidden_gradients + rcx * 8 - 8], xmm0
    loop .clip_loop
    
    leave
    ret

initialize_batch:
    ; Initialize batch processing
    push rbp
    mov rbp, rsp
    
    ; Zero out batch gradients
    mov rcx, BATCH_SIZE * HIDDEN_SIZE
    xor rax, rax
.zero_loop:
    mov qword [batch_gradients + rcx * 8 - 8], rax
    loop .zero_loop
    
    leave
    ret