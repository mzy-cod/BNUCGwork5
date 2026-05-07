import taichi as ti

# 初始化 Taichi，注意这里已经修正了之前版本的 ti.math 问题
ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=float, shape=(res_x, res_y))

light_pos = ti.Vector.field(3, dtype=float, shape=())
max_bounces = ti.field(dtype=int, shape=())
spp = ti.field(dtype=int, shape=()) # 新增：每个像素的采样数 (Samples Per Pixel)

light_pos[None] = [2.0, 4.0, 3.0]
max_bounces[None] = 4 # 增加弹射次数以让光线能穿透玻璃
spp[None] = 4         # 默认开启 4x MSAA

@ti.func
def intersect_sphere(ro, rd, center, radius):
    oc = ro - center
    b = 2.0 * rd.dot(oc)
    c = oc.dot(oc) - radius * radius
    discriminant = b * b - 4.0 * c
    
    hit = False
    t = 1e20
    if discriminant > 0:
        sqrt_d = ti.sqrt(discriminant)
        t1 = (-b - sqrt_d) / 2.0
        t2 = (-b + sqrt_d) / 2.0
        if t1 > 1e-4:
            t = t1
            hit = True
        elif t2 > 1e-4:
            t = t2
            hit = True
    return hit, t

@ti.func
def intersect_plane(ro, rd, plane_y):
    hit = False
    t = 1e20
    if ti.abs(rd.y) > 1e-4:
        tp = -(ro.y - plane_y) / rd.y
        if tp > 1e-4:
            t = tp
            hit = True
    return hit, t

@ti.func
def scene_intersect(ro, rd):
    t_min = 1e20
    # 0=天空, 1=平面, 2=玻璃球, 3=银色镜面球
    mat_id = 0 
    hit_p = ti.Vector([0.0, 0.0, 0.0])
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    diffuse_color = ti.Vector([0.0, 0.0, 0.0])

    # 1. 棋盘格平面
    hit, t = intersect_plane(ro, rd, -1.0)
    if hit and t < t_min:
        t_min = t
        mat_id = 1
        hit_p = ro + rd * t
        hit_n = ti.Vector([0.0, 1.0, 0.0])
        grid_x = int(ti.floor(hit_p.x * 2.0))
        grid_z = int(ti.floor(hit_p.z * 2.0))
        if (grid_x + grid_z) % 2 == 0:
            diffuse_color = ti.Vector([0.9, 0.9, 0.9])
        else:
            diffuse_color = ti.Vector([0.1, 0.1, 0.1])

    # 2. 【修改】玻璃球 (Glass Sphere) - 位于左侧
    center_glass = ti.Vector([-1.5, 0.0, 0.0])
    hit, t = intersect_sphere(ro, rd, center_glass, 1.0)
    if hit and t < t_min:
        t_min = t
        mat_id = 2 
        hit_p = ro + rd * t
        hit_n = (hit_p - center_glass).normalized()

    # 3. 银色镜面球
    center_mirror = ti.Vector([1.5, 0.0, 0.0])
    hit, t = intersect_sphere(ro, rd, center_mirror, 1.0)
    if hit and t < t_min:
        t_min = t
        mat_id = 3
        hit_p = ro + rd * t
        hit_n = (hit_p - center_mirror).normalized()

    return mat_id, t_min, hit_p, hit_n, diffuse_color

@ti.func
def refract_ray(rd, n, ior):
    """
    计算折射光线方向与全反射判断
    rd: 入射光方向, n: 表面法线, ior: 材质折射率 (假设外部是空气 ior=1.0)
    """
    cos_i = rd.dot(n)
    eta = 1.0 / ior  # 默认：从空气进入介质
    out_n = n
    
    if cos_i > 0.0: 
        # cos_i > 0 说明入射光线和法线同向，即光线正从介质内部射向外部
        eta = ior / 1.0
        out_n = -n    # 翻转法线，使其指向内部
    else:
        # 正常从外部射入内部
        cos_i = -cos_i
        
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    
    is_tir = False
    refr_dir = ti.Vector([0.0, 0.0, 0.0])
    
    if k < 0.0:
        is_tir = True # 根号下为负，发生全反射 (Total Internal Reflection)
    else:
        # 斯涅尔定律向量形式
        refr_dir = eta * rd + (eta * cos_i - ti.sqrt(k)) * out_n
        
    return is_tir, refr_dir.normalized(), out_n

@ti.func
def trace(ro, rd):
    """
    为了支持 MSAA，将单根光线的追踪逻辑抽离为一个函数
    """
    final_color = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])

    current_ro = ro
    current_rd = rd

    for _ in range(max_bounces[None]):
        mat_id, t, hit_p, hit_n, diff_c = scene_intersect(current_ro, current_rd)

        if mat_id == 0: # 天空
            bg_color = ti.Vector([0.05, 0.1, 0.15])
            final_color += throughput * bg_color
            break

        if mat_id == 1: # 漫反射平面
            light_dir = light_pos[None] - hit_p
            dist_to_light = light_dir.norm()
            light_dir = light_dir.normalized()

            shadow_ro = hit_p + hit_n * 1e-4 
            shadow_mat, shadow_t, dummy_p, dummy_n, dummy_c = scene_intersect(shadow_ro, light_dir)

            in_shadow = False
            # 玻璃球（mat_id=2）会透光，所以如果阴影射线打到玻璃，不应完全判定为阴影（为了简化这里做纯硬阴影判断，可自行改进半透明阴影）
            if shadow_mat != 0 and shadow_mat != 2 and shadow_t < dist_to_light:
                in_shadow = True

            ambient = ti.Vector([0.1, 0.1, 0.1]) * diff_c
            if in_shadow:
                final_color += throughput * ambient
            else:
                ndotl = ti.max(0.0, hit_n.dot(light_dir))
                diffuse = diff_c * ndotl
                final_color += throughput * (ambient + diffuse)
            break 

        elif mat_id == 3: # 镜面
            current_rd = current_rd - 2.0 * current_rd.dot(hit_n) * hit_n
            current_rd = current_rd.normalized()
            current_ro = hit_p + hit_n * 1e-4 
            throughput *= 0.8 
            continue 

        elif mat_id == 2: # 【新增】玻璃材质
            # 玻璃折射率通常在 1.5 左右
            is_tir, refr_dir, out_n = refract_ray(current_rd, hit_n, 1.5)
            
            if is_tir:
                # 发生全反射，按镜面反射处理
                current_rd = current_rd - 2.0 * current_rd.dot(out_n) * out_n
                current_ro = hit_p + out_n * 1e-4
            else:
                # 正常折射
                current_rd = refr_dir
                # 注意：折射光线是穿过表面的，因此起点的偏移量方向是 -out_n
                current_ro = hit_p - out_n * 1e-4
            
            # 玻璃吸收少许光线，透射率很高
            throughput *= 0.95
            continue

    return final_color

@ti.kernel
def render():
    for i, j in pixels:
        cam_pos = ti.Vector([0.0, 1.0, 5.0])
        look_at = ti.Vector([0.0, -0.5, 0.0])
        forward = (look_at - cam_pos).normalized()
        right = forward.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
        up = right.cross(forward).normalized()

        pixel_color = ti.Vector([0.0, 0.0, 0.0])
        
        # 【新增】MSAA 抗锯齿：在一个像素内随机采样多次
        for s in range(spp[None]):
            # 加上 ti.random() 实现子像素级别的随机偏移
            u = (i + ti.random()) / res_x * 2.0 - 1.0
            v = (j + ti.random()) / res_y * 2.0 - 1.0
            v *= res_y / res_x 

            ro = cam_pos
            rd = (forward + u * right + v * up).normalized()
            
            pixel_color += trace(ro, rd)
            
        # 求平均颜色并钳制
        pixel_color /= float(spp[None])
        pixels[i, j] = ti.max(0.0, ti.min(1.0, pixel_color))

def main():
    window = ti.ui.Window("Advanced Ray Tracing (Glass & MSAA)", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        render() 
        canvas.set_image(pixels) 

        with gui.sub_window("Controls", 0.65, 0.05, 0.32, 0.3):
            lx = gui.slider_float("Light X", light_pos[None].x, -10.0, 10.0)
            ly = gui.slider_float("Light Y", light_pos[None].y, 0.0, 10.0)
            lz = gui.slider_float("Light Z", light_pos[None].z, -10.0, 10.0)
            light_pos[None] = ti.Vector([lx, ly, lz])
            
            max_bounces[None] = gui.slider_int("Max Bounces", max_bounces[None], 1, 8)
            spp[None] = gui.slider_int("AA Samples (SPP)", spp[None], 1, 16)

        window.show()

if __name__ == "__main__":
    main()