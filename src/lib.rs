use wgpu::{PipelineCompilationOptions, RenderPipelineDescriptor, PipelineLayoutDescriptor, COPY_BUFFER_ALIGNMENT, VertexBufferLayout, DepthStencilState, MultisampleState, BufferDescriptor, RenderPipeline, PrimitiveState, VertexStepMode, FragmentState, TextureFormat, BufferAddress, BufferUsages, IndexFormat, VertexState, RenderPass, Buffer, Device, Queue};

use lyon_tessellation::{
    FillVertexConstructor,
    FillTessellator,
    FillOptions,
    FillBuilder,
    FillVertex,
    BuffersBuilder,
    VertexBuffers,
};

type Bound = (u32, u32, u32, u32);

pub struct Shape {
    pub constructor: Box<dyn Fn(&mut FillBuilder)>,
    pub bound: Bound
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
    z: f32
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3, 2 => Float32];

    fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[derive(Clone)]
pub struct VertexConstructor;
impl FillVertexConstructor<Vertex> for VertexConstructor {
    fn new_vertex(&mut self, mut vertex: FillVertex) -> Vertex {
        let attrs: [f32; 4] = vertex.interpolated_attributes().try_into()
            .expect("Expected builder attributes to be 3 f32's representing RGB color values. And one f32 representing zindex");
        Vertex{
            position: vertex.position().to_array(),
            color: [attrs[0], attrs[1], attrs[2]],
            z: attrs[3]
        }
    }
}

pub struct LyonRenderer {
    render_pipeline: RenderPipeline,
    vertex_buffer_size: u64,
    vertex_buffer: Buffer,
    index_buffer_size: u64,
    index_buffer: Buffer,
    lyon_buffers: VertexBuffers<Vertex, u16>,
    shape_buffer: Vec<(usize, usize, Bound)>
}

impl LyonRenderer {
    /// Create all unchanging resources here.
    pub fn new(
        device: &Device,
        texture_format: &TextureFormat,
        multisample: MultisampleState,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor::default());
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[
                    Vertex::layout()
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
            label: None,
            size: vertex_buffer_size,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer_size = Self::next_copy_buffer_size(4096);
        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: index_buffer_size,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lyon_buffers: VertexBuffers<Vertex, u16> = VertexBuffers::new();
        LyonRenderer{
            render_pipeline,
            vertex_buffer_size,
            vertex_buffer,
            index_buffer_size,
            index_buffer,
            lyon_buffers,
            shape_buffer: Vec::new()
        }
    }

    /// Prepare for rendering this frame; create all resources that will be
    /// used during the next render that do not already exist.
    pub fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        fill_options: &FillOptions,
        shapes: Vec<Shape>
    ) {
        self.lyon_buffers.clear();
        self.shape_buffer.clear();

        let mut index = 0;

        let mut buffer = BuffersBuilder::new(&mut self.lyon_buffers, VertexConstructor);
        let mut tessellator = FillTessellator::new();
        for shape in shapes {
            let mut builder = tessellator.builder_with_attributes(4, fill_options, &mut buffer);
            (shape.constructor)(&mut builder);
            builder.build().unwrap();

            let buffer_len = buffer.buffers().indices.len();
            self.shape_buffer.push((index, buffer_len, shape.bound));
            index = buffer_len;
        }

        if self.lyon_buffers.vertices.is_empty() || self.lyon_buffers.indices.is_empty() {return;}

        let vertices_raw = bytemuck::cast_slice(&self.lyon_buffers.vertices);
        if self.vertex_buffer_size >= vertices_raw.len() as u64 {
            Self::write_buffer(queue, &self.vertex_buffer, vertices_raw);
        } else {
            let (vertex_buffer, vertex_buffer_size) = Self::create_oversized_buffer(
                device, None, vertices_raw, BufferUsages::VERTEX | BufferUsages::COPY_DST
            );
            self.vertex_buffer = vertex_buffer;
            self.vertex_buffer_size = vertex_buffer_size;
        }

        let indices_raw = bytemuck::cast_slice(&self.lyon_buffers.indices);
        if self.index_buffer_size >= indices_raw.len() as u64 {
            Self::write_buffer(queue, &self.index_buffer, indices_raw);
        } else {
            let (index_buffer, index_buffer_size) = Self::create_oversized_buffer(
                device, None, indices_raw, BufferUsages::INDEX | BufferUsages::COPY_DST
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
        for (start, end, bound) in &self.shape_buffer {
            render_pass.set_scissor_rect(bound.0, bound.1, bound.2, bound.3);
            render_pass.draw_indexed(*start as u32..*end as u32, 0, 0..1);
        }
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
