Random.seed!(42)
# A deterministic hybrid backbone with production-sized Delta head dimensions.
function synthetic_backbone(; hidden=256, intermediate=512, depth=4)
    cfg = (hidden=hidden, heads=4, kv_heads=2, head_dim=64, key_heads=2,
           value_heads=4, key_dim=128, value_dim=128, eps=1f-6,
           rope_theta=10000f0, rotary_dim=16)
    weight(a,b) = randn(Float32,a,b) ./ sqrt(Float32(a))
    layers = map(1:depth) do i
        attention = if i % 4 == 0
            (kind=:full, q=weight(hidden,512), k=weight(hidden,128),
             v=weight(hidden,128), out=weight(256,hidden),
             q_norm=zeros(Float32,64), k_norm=zeros(Float32,64))
        else
            (kind=:delta, qkv=weight(hidden,1024), z=weight(hidden,512),
             a=weight(hidden,4), b=weight(hidden,4),
             conv=transpose(randn(Float32,1024,4) .* 0.1f0),
             a_decay=-ones(Float32,4), dt_bias=zeros(Float32,4),
             norm=ones(Float32,128), out=weight(512,hidden))
        end
        attention = QDC.native_attention_weights(Val(:cpu), attention)
        mlp=QDC.native_mlp_weights(Val(:cpu), weight(hidden,intermediate),
             weight(hidden,intermediate), weight(intermediate,hidden))
        QDC.QwenLayer(attention,mlp,zeros(Float32,hidden),zeros(Float32,hidden))
    end
    concrete = Vector{Union{unique(typeof.(layers))...}}(layers)
    QwenBackbone(weight(hidden,256),concrete,zeros(Float32,hidden),cfg)
end

function device_backbone(b, device)
    d = Val(device)
    arr(x) = QDC.native_array(d, x)
    layers = map(b.layers) do layer
        attention = map(x -> x isa AbstractArray ? arr(x) : x, layer.attention)
        # Synthetic convolution has the CPU's transposed storage wrapper.
        attention = QDC.native_attention_weights(d, attention)
        mlp = QDC.native_mlp_weights(d, arr(layer.mlp.gate), arr(layer.mlp.up), arr(layer.mlp.down))
        QDC.QwenLayer(attention, mlp, arr(layer.input_norm), arr(layer.post_norm))
    end
    QwenBackbone(arr(b.embedding), Vector{Union{unique(typeof.(layers))...}}(layers), arr(b.final_norm), b.config)
end
