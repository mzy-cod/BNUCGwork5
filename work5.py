import taichi as ti

# 1. 初始化 Taichi，指定使用 GPU 作为后端
ti.init(arch=ti.gpu)

# 画面分辨率
res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=float, shape=(res_x, res_y))

# 全局变量：光源位置和最大弹射次数 (Task 4 UI交互需要)
light_pos = ti.Vector.field(3, dtype=float, shape=())
max_bounces = ti.field(dtype=int, shape=())

# 初始化默认值
light_pos[None] = [2.0, 4.0, 3.0]
max_bounces[None] = 3

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """
    计算光线与球体的交点 (一元二次方程)
    """
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
        # 忽略极小值，避免自交
        if t1 > 1e-4:
            t = t1
            hit = True
        elif t2 > 1e-4:
            t = t2
            hit = True
    return hit, t

@ti.func
def intersect_plane(ro, rd, plane_y):
    """
    计算光线与无限大水平面的交点
    """
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
    """
    任务 1：搭建包含平面的三维场景 (隐式几何体定义)
    遍历场景中所有物体，返回最近的交点信息
    """
    t_min = 1e20
    # 材质 ID 区分：0=天空(无击中), 1=漫反射平面, 2=红色漫反射球, 3=银色镜面球
    mat_id = 0 
    hit_p = ti.Vector([0.0, 0.0, 0.0])
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    diffuse_color = ti.Vector([0.0, 0.0, 0.0])

    # 1. 无限大平面 (Ground Plane)
    hit, t = intersect_plane(ro, rd, -1.0)
    if hit and t < t_min:
        t_min = t
        mat_id = 1
        hit_p = ro + rd * t
        hit_n = ti.Vector([0.0, 1.0, 0.0])
        # 实现黑白棋盘格纹理 (奇偶性判断)
        grid_x = int(ti.floor(hit_p.x * 2.0))
        grid_z = int(ti.floor(hit_p.z * 2.0))
        if (grid_x + grid_z) % 2 == 0:
            diffuse_color = ti.Vector([0.9, 0.9, 0.9])
        else:
            diffuse_color = ti.Vector([0.1, 0.1, 0.1])

    # 2. 红色漫反射球 (Red Diffuse Sphere)
    center_red = ti.Vector([-1.5, 0.0, 0.0])
    hit, t = intersect_sphere(ro, rd, center_red, 1.0)
    if hit and t < t_min:
        t_min = t
        mat_id = 2
        hit_p = ro + rd * t
        hit_n = (hit_p - center_red).normalized()
        diffuse_color = ti.Vector([0.8, 0.05, 0.05])

    # 3. 银色镜面球 (Silver Mirror Sphere)
    center_mirror = ti.Vector([1.5, 0.0, 0.0])
    hit, t = intersect_sphere(ro, rd, center_mirror, 1.0)
    if hit and t < t_min:
        t_min = t
        mat_id = 3
        hit_p = ro + rd * t
        hit_n = (hit_p - center_mirror).normalized()
        # 镜面无漫反射颜色

    return mat_id, t_min, hit_p, hit_n, diffuse_color

@ti.kernel
def render():
    """
    核心渲染 Kernel
    """
    for i, j in pixels:
        # 简易摄像机设置
        u = (i + 0.5) / res_x * 2.0 - 1.0
        v = (j + 0.5) / res_y * 2.0 - 1.0
        v *= res_y / res_x # 适应长宽比

        # 摄像机位于原点上方，看向正前方
        cam_pos = ti.Vector([0.0, 1.0, 5.0])
        look_at = ti.Vector([0.0, -0.5, 0.0])
        forward = (look_at - cam_pos).normalized()
        right = forward.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
        up = right.cross(forward).normalized()

        # 生成主光线
        ro = cam_pos
        rd = (forward + u * right + v * up).normalized()

        # 任务 2：实现基于迭代的光线弹射
        final_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])

        current_ro = ro
        current_rd = rd

        for bounce in range(max_bounces[None]):
            mat_id, t, hit_p, hit_n, diff_c = scene_intersect(current_ro, current_rd)

            # 如果没打到任何物体（打到天空）
            if mat_id == 0:
                bg_color = ti.Vector([0.05, 0.1, 0.15]) # 简单深色背景
                final_color += throughput * bg_color
                break

            # 材质分支处理
            if mat_id == 3: 
                # 是镜面材质 (Mirror)
                # 反射公式：R = L - 2(L·N)N
                current_rd = current_rd - 2.0 * current_rd.dot(hit_n) * hit_n
                current_rd = current_rd.normalized()
                
                # 任务 3：核心避坑点，沿着法线外偏一个极小值避免自交
                current_ro = hit_p + hit_n * 1e-4 
                
                throughput *= 0.8 # 乘上反射率衰减
                continue # 继续下一次弹射循环

            else: 
                # 是漫反射材质 (Diffuse 平面或红球)
                light_dir = light_pos[None] - hit_p
                dist_to_light = light_dir.norm()
                light_dir = light_dir.normalized()

                # 任务 3：实现硬阴影，向光源发射一条暗影射线
                # 沿着法线偏移防止 Shadow Acne
                shadow_ro = hit_p + hit_n * 1e-4 
                shadow_mat, shadow_t, _, _, _ = scene_intersect(shadow_ro, light_dir)

                # 判断该点是否在阴影中 (击中物体且在到达光源前)
                in_shadow = False
                if shadow_mat != 0 and shadow_t < dist_to_light:
                    in_shadow = True

                # 环境光
                ambient = ti.Vector([0.1, 0.1, 0.1]) * diff_c
                
                if in_shadow:
                    # 阴影中仅保留环境光
                    final_color += throughput * ambient
                else:
                    # 不在阴影中，计算 Phong 模型 (此处简化为 Lambertian 漫反射)
                    ndotl = ti.max(0.0, hit_n.dot(light_dir))
                    diffuse = diff_c * ndotl
                    final_color += throughput * (ambient + diffuse)

                # Whitted-Style：击中漫反射物体后，计算当前光照颜色并 break 终止该条光线传播
                break 

        # 钳制颜色值以防越界
        pixels[i, j] = ti.max(0.0, ti.min(1.0, final_color))

def main():
    # 任务 4：完成 UI 交互面板
    window = ti.ui.Window("Whitted-Style Ray Tracing Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        render() # 调用渲染函数
        canvas.set_image(pixels) # 将渲染结果赋值给画布

        # 构建滑动条控件窗口
        with gui.sub_window("Controls", 0.70, 0.05, 0.28, 0.25):
            # 动态改变点光源的三维坐标
            lx = gui.slider_float("Light X", light_pos[None].x, -10.0, 10.0)
            ly = gui.slider_float("Light Y", light_pos[None].y, 0.0, 10.0)
            lz = gui.slider_float("Light Z", light_pos[None].z, -10.0, 10.0)
            light_pos[None] = ti.Vector([lx, ly, lz])
            
            # 最大弹射次数：观察其变化带来的差异
            max_bounces[None] = gui.slider_int("Max Bounces", max_bounces[None], 1, 5)

        window.show()

if __name__ == "__main__":
    main()