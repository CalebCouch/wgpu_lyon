use wgpu::{
    PipelineCompilationOptions,
    RenderPipelineDescriptor,
    PipelineLayoutDescriptor,
    COPY_BUFFER_ALIGNMENT,
    VertexBufferLayout,
    DepthStencilState,
    MultisampleState,
    BufferDescriptor,
    RenderPipeline,
    PrimitiveState,
    VertexStepMode,
    FragmentState,
    TextureFormat,
    BufferAddress,
    ShaderModule,
    BufferUsages,
    IndexFormat,
    VertexState,
    RenderPass,
    Buffer,
    Device,
    Queue,
};

use lyon_tessellation::{
    FillVertexConstructor,
    FillTessellator,
    FillOptions,
    FillBuilder,
    FillVertex,
    BuffersBuilder,
    VertexBuffers,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DefaultVertex {
    position: [f32; 2],
    color: [f32; 3],
}

impl DefaultVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3];
}

impl Vertex for DefaultVertex {
    type Constructor = DefaultVertexConstructor;

    fn constructor() -> Self::Constructor {DefaultVertexConstructor}
    fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }

    fn shader(device: &Device) -> ShaderModule {
        device.create_shader_module(wgpu::include_wgsl!("default_shader.wgsl"))
    }
}

#[derive(Clone)]
pub struct DefaultVertexConstructor;
impl FillVertexConstructor<DefaultVertex> for DefaultVertexConstructor {
    fn new_vertex(&mut self, mut vertex: FillVertex) -> DefaultVertex {
        DefaultVertex{
            position: vertex.position().to_array(),
            color: vertex.interpolated_attributes().try_into()
            .expect("Expected builder attributes to be 3 f32's representing RGB color values.")
        }
    }
}

pub trait Vertex: Copy + bytemuck::Pod + bytemuck::Zeroable {
    type Constructor: FillVertexConstructor<Self>;

    fn constructor() -> Self::Constructor;
    fn layout() -> VertexBufferLayout<'static>;
    fn shader(device: &Device) -> ShaderModule;
}

pub struct LyonRenderer<V: Vertex = DefaultVertex> {
    render_pipeline: RenderPipeline,
    vertex_buffer_size: u64,
    vertex_buffer: Buffer,
    index_buffer_size: u64,
    index_buffer: Buffer,
    lyon_buffers: VertexBuffers<V, u16>,
}

impl<V: Vertex> LyonRenderer<V> {
    /// Create all unchanging resources here.
    pub fn new(
        device: &Device,
        texture_format: &TextureFormat,
        multisample: MultisampleState,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {

        let shader = V::shader(device);
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor::default());
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[
                    V::layout()
                ]
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some((*texture_format).into())],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil,
            multisample,
            multiview: None,
            cache: None
        });

        let vertex_buffer_size = Self::next_copy_buffer_size(4096);
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Lyon Vertex Buffer"),
            size: vertex_buffer_size,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer_size = Self::next_copy_buffer_size(4096);
        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Lyon Index Buffer"),
            size: index_buffer_size,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lyon_buffers: VertexBuffers<V, u16> = VertexBuffers::new();
        LyonRenderer{
            render_pipeline,
            vertex_buffer_size,
            vertex_buffer,
            index_buffer_size,
            index_buffer,
            lyon_buffers,
        }
    }

    /// Prepare for rendering this frame; create all resources that will be
    /// used during the next render that do not already exist.
    pub fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        fill_options: &FillOptions,
        callbacks: Vec<impl Fn(&mut FillBuilder)>
    ) {
        self.lyon_buffers.clear();

        let mut buffer = BuffersBuilder::new(&mut self.lyon_buffers, V::constructor());

        let mut tessellator = FillTessellator::new();
        for callback in callbacks {
            let mut builder = tessellator.builder_with_attributes(3, fill_options, &mut buffer);

            callback(&mut builder);

            builder.build().unwrap();
        }

        if self.lyon_buffers.vertices.is_empty() || self.lyon_buffers.indices.is_empty() {return;}

        let vertices_raw = bytemuck::cast_slice(&self.lyon_buffers.vertices);
        if self.vertex_buffer_size >= vertices_raw.len() as u64 {
            Self::write_buffer(queue, &self.vertex_buffer, vertices_raw);
        } else {
            let (vertex_buffer, vertex_buffer_size) = Self::create_oversized_buffer(
                device,
                Some("Lyon Vertex Buffer"),
                vertices_raw,
                BufferUsages::VERTEX | BufferUsages::COPY_DST
            );
            self.vertex_buffer = vertex_buffer;
            self.vertex_buffer_size = vertex_buffer_size;
        }

        let indices_raw = bytemuck::cast_slice(&self.lyon_buffers.indices);
        if self.index_buffer_size >= indices_raw.len() as u64 {
            Self::write_buffer(queue, &self.index_buffer, indices_raw);
        } else {
            let (index_buffer, index_buffer_size) = Self::create_oversized_buffer(
                device,
                Some("Lyon Index Buffer"),
                indices_raw,
                BufferUsages::INDEX | BufferUsages::COPY_DST
            );
            self.index_buffer = index_buffer;
            self.index_buffer_size = index_buffer_size;
        }
    }

    /// Render using caller provided render pass.
    pub fn render(&self, render_pass: &mut RenderPass<'_>) {
        if self.lyon_buffers.vertices.is_empty() || self.lyon_buffers.indices.is_empty() {return;}

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.lyon_buffers.indices.len() as u32, 0, 0..1);
    }

    fn write_buffer(queue: &Queue, buffer: &Buffer, slice: &[u8]) {
        let pad: usize = slice.len() % 4;
        let slice = if pad != 0 {
            &[slice, &vec![0u8; pad]].concat()
        } else {slice};
        queue.write_buffer(buffer, 0, slice);
    }

    fn next_copy_buffer_size(size: u64) -> u64 {
        let align_mask = COPY_BUFFER_ALIGNMENT - 1;
        ((size.next_power_of_two() + align_mask) & !align_mask).max(COPY_BUFFER_ALIGNMENT)
    }

    fn create_oversized_buffer(
        device: &Device,
        label: Option<&str>,
        contents: &[u8],
        usage: BufferUsages,
    ) -> (Buffer, u64) {
        let size = Self::next_copy_buffer_size(contents.len() as u64);
        let buffer = device.create_buffer(&BufferDescriptor {
            label,
            size,
            usage,
            mapped_at_creation: true,
        });
        buffer.slice(..).get_mapped_range_mut()[..contents.len()].copy_from_slice(contents);
        buffer.unmap();
        (buffer, size)
    }
}
